"""
Visualisation helper for PyTheus-style photonic graphs.

Standalone module (does not modify pytheus itself, and never touches the
underlying optimisation data -- it only reads weight vectors and draws them).
Reuses the same circular vertex layout and edge-colour convention as
``pytheus.graphplot.graphPlot`` so figures look consistent with the rest of
the project, but adds:

- real weight labels on every edge (pytheus's own label-drawing code is
  present but commented out, so it never renders text)
- smooth curved (quadratic-Bezier) edges so parallel/overlapping edges
  between the same vertex pair stay visually separable
- only active edges (|weight| > threshold) are drawn; missing/pruned edges
  are omitted entirely
- a "topology only" mode (no weight labels, uniform edge style) plus a
  split-by-internal-mode 2x2 panel view, for dense graphs where putting
  every weight on one figure becomes unreadable
- a companion weight-table exporter (JSON/CSV) for traceability when labels
  are left off the figure
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pytheus import theseus as th

EDGE_COLORS = ["dodgerblue", "firebrick", "limegreen", "darkorange", "purple", "yellow", "cyan"]


def vertex_positions(n: int, rotation: float = 0.0, clockwise: bool = False) -> Dict[int, Tuple[float, float]]:
    """rotation (radians) rotates the whole layout, e.g. rotation=np.pi/4 turns a
    4-node N/E/S/W diamond into a NE/NW/SW/SE square so cycle edges (0-1-2-3-0)
    align horizontally/vertically instead of diagonally.
    clockwise=True numbers vertices clockwise instead of the default
    counter-clockwise. E.g. for n=4, rotation=3*np.pi/4, clockwise=True puts
    node 0 at top-left and numbers 1/2/3 clockwise (top-right, bottom-right,
    bottom-left)."""
    step = np.linspace(0, 2 * np.pi * (n - 1) / n, n)
    if clockwise:
        step = -step
    angles = step + rotation
    rad = 0.9
    return {i: (rad * np.cos(a), rad * np.sin(a)) for i, a in enumerate(angles)}


def edge_table(
    edges: Sequence[Tuple[int, int, int, int]],
    weights: Sequence[float],
    weight_threshold: float = 0.0,
    iteration: Union[int, str, None] = None,
) -> List[dict]:
    """Return the traceability table: edge_index, edge_tuple, weight, active[, iteration]."""
    rows = []
    for idx, (edge, w) in enumerate(zip(edges, weights)):
        row = {
            "edge_index": idx,
            "edge_tuple": tuple(int(v) for v in edge),
            "weight": float(w),
            "active": bool(abs(w) > weight_threshold),
        }
        if iteration is not None:
            row["iteration"] = iteration
        rows.append(row)
    return rows


def save_weight_table(
    edges: Sequence[Tuple[int, int, int, int]],
    weights: Sequence[float],
    weight_threshold: float = 0.0,
    iteration: Union[int, str, None] = None,
    json_path: Union[str, Path, None] = None,
    csv_path: Union[str, Path, None] = None,
) -> List[dict]:
    """Write the full edge/weight table (companion file for label-free figures)."""
    rows = edge_table(edges, weights, weight_threshold=weight_threshold, iteration=iteration)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump({"iteration": iteration, "weight_threshold": weight_threshold, "edges": rows}, f, indent=2)

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            header = ["edge_index", "edge_tuple", "weight", "active"]
            if iteration is not None:
                header.append("iteration")
            writer.writerow(header)
            for row in rows:
                line = [row["edge_index"], row["edge_tuple"], row["weight"], row["active"]]
                if iteration is not None:
                    line.append(row["iteration"])
                writer.writerow(line)

    return rows


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _draw_edges(
    ax,
    edges: Sequence[Tuple[int, int, int, int]],
    weights: Sequence[float],
    verts: Dict[int, Tuple[float, float]],
    weight_threshold: float,
    curvature: float,
    multi_edge_spacing: float,
    show_labels: bool,
    uniform_style: bool,
    label_fontsize: float = 15.0,
    linewidth_scale: float = 1.0,
) -> bool:
    """Draw curved active edges (and, if show_labels, their weight labels) onto `ax`.

    `edges`/`weights` may be a subset (e.g. one colour family) -- multiplicity
    and spacing are computed from whatever is passed in. Returns True if any
    drawn edge had a negative weight (used to decide whether to show the
    "negative weight" legend entry).
    """
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    bleached = th.edgeBleach(edges)  # {(v1, v2): [(c1, c2), ...]}

    curve_t_half1 = np.linspace(0.0, 0.5, 24)
    curve_t_half2 = np.linspace(0.5, 1.0, 24)

    any_negative = False

    for (v1, v2), colorings in bleached.items():
        mult = len(colorings)
        vert1 = np.array(verts[int(v1)])
        vert2 = np.array(verts[int(v2)])
        diff = vert1 - vert2
        rect = np.array([diff[1], -diff[0]])
        rect /= np.linalg.norm(rect)
        mid = (vert1 + vert2) / 2

        for ind, (c1, c2) in enumerate(colorings):
            idx = edge_to_idx[(int(v1), int(v2), int(c1), int(c2))]
            w = weights[idx]
            if abs(w) <= weight_threshold:
                continue  # inactive/pruned edge -- not drawn at all

            offset = (2 * ind - mult + 1) * multi_edge_spacing * curvature
            ctrl = mid + offset * rect

            pts1 = _quadratic_bezier(vert1, ctrl, vert2, curve_t_half1)
            pts2 = _quadratic_bezier(vert1, ctrl, vert2, curve_t_half2)
            curve_mid = pts1[-1]  # == pts2[0], true midpoint of the curve

            col1 = EDGE_COLORS[int(c1) % len(EDGE_COLORS)]
            col2 = EDGE_COLORS[int(c2) % len(EDGE_COLORS)]

            if uniform_style:
                lw, alpha = 2.6, 0.85
            else:
                lw = 1.4 + 5.0 * min(abs(w), 1.0)
                alpha = 0.45 + 0.55 * min(abs(w), 1.0)
            lw *= linewidth_scale

            ax.plot(pts1[:, 0], pts1[:, 1], color=col1, linewidth=lw, alpha=alpha,
                     solid_capstyle="round", zorder=5)
            ax.plot(pts2[:, 0], pts2[:, 1], color=col2, linewidth=lw, alpha=alpha,
                     solid_capstyle="round", zorder=5)

            if show_labels:
                if w < 0:
                    any_negative = True
                    ax.plot(
                        curve_mid[0], curve_mid[1], marker="d", markersize=15, markeredgewidth=1.6,
                        markeredgecolor="black", color="white", zorder=6,
                    )
                ax.text(
                    curve_mid[0], curve_mid[1], f"{w:+.2f}", fontsize=label_fontsize, color="black",
                    fontweight="bold", ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="dimgray", lw=0.6, alpha=0.92),
                )
            elif w < 0:
                any_negative = True

    return any_negative


def _draw_nodes(ax, verts: Dict[int, Tuple[float, float]], node_radius: float = 0.15, node_fontsize: float = 19.0) -> None:
    for i, (x, y) in verts.items():
        ax.add_patch(
            plt.Circle((x, y), node_radius, facecolor="lightgrey", edgecolor="dimgray", linewidth=2.2, zorder=10)
        )
        ax.text(x, y, str(i), fontsize=node_fontsize, ha="center", va="center", zorder=11, fontweight="bold")


def _finalize_ax(ax, node_radius: float = 0.15, title: str = "", title_fontsize: float = 15.0) -> None:
    lim = 1.0 + node_radius + 0.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize)


def plot_pytheus_graph(
    edges: Sequence[Tuple[int, int, int, int]],
    weights: Sequence[float],
    n: int,
    title: str = "",
    out_stub: Union[str, Path, None] = None,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    figsize: Optional[float] = None,
    curvature: Optional[float] = None,
    weight_threshold: float = 0.0,
    multi_edge_spacing: float = 0.75,
    show_labels: bool = True,
    uniform_style: Optional[bool] = None,
    linewidth_scale: float = 1.0,
    node_fontsize: float = 19.0,
    rotation: float = 0.0,
    clockwise: bool = False,
    show: bool = False,
):
    """
    Draw a PyTheus-style graph with curved edges.

    edges   : full edge catalogue, e.g. pytheus.theseus.buildAllEdges(...)
    weights : matching weight vector (same length/order as `edges`)
    weight_threshold : an edge counts as active iff abs(weight) > threshold
                        (use 0.0 for a strict nonzero check). Only active
                        edges are drawn.
    figsize, curvature : if left as None, both auto-scale with the number of
                          active edges so dense graphs stay readable.
    multi_edge_spacing  : scales how far apart parallel edges between the
                          same vertex pair are spread (< 1 pulls them closer
                          together, keeping them grouped).
    show_labels  : if False, produces a clean "topology only" figure with no
                   weight text and no negative-weight markers (uniform edge
                   style is used automatically unless `uniform_style` is set
                   explicitly).
    linewidth_scale : multiplier applied to every edge's line width (both
                       uniform-style and weight-scaled lines). 1.0 = default
                       thickness; use e.g. 2.0 for edges twice as thick.
    node_fontsize   : font size of the vertex-index numbers drawn inside each
                      node circle. Default 19.0.
    rotation : rotates the whole vertex layout, in radians. E.g. for n=4,
               rotation=np.pi/4 turns the default N/E/S/W diamond into a
               NE/NW/SW/SE square, so cycle edges (0-1-2-3-0) align
               horizontally/vertically instead of diagonally.
    clockwise : if True, numbers vertices clockwise instead of the default
                counter-clockwise. E.g. for n=4, rotation=3*np.pi/4,
                clockwise=True puts node 0 at top-left and numbers 1/2/3
                clockwise (top-right, bottom-right, bottom-left).

    Returns (rows, saved_paths) where `rows` is the full traceability table
    (all edges, active and inactive) produced by `edge_table`.
    """
    edges = [tuple(int(v) for v in e) for e in edges]
    weights = [float(w) for w in weights]
    assert len(edges) == len(weights), "edges and weights must be the same length"

    if uniform_style is None:
        uniform_style = not show_labels

    active_count = sum(1 for w in weights if abs(w) > weight_threshold)

    if figsize is None:
        figsize = float(np.clip(9.5 + 0.22 * active_count, 10.5, 17.0))
    if curvature is None:
        curvature = float(np.clip(0.11 + 0.006 * active_count, 0.11, 0.30))

    verts = vertex_positions(n, rotation=rotation, clockwise=clockwise)

    fig, ax = plt.subplots(figsize=(figsize, figsize))

    any_negative = _draw_edges(
        ax, edges, weights, verts,
        weight_threshold=weight_threshold, curvature=curvature,
        multi_edge_spacing=multi_edge_spacing, show_labels=show_labels,
        uniform_style=uniform_style, linewidth_scale=linewidth_scale,
    )
    _draw_nodes(ax, verts, node_fontsize=node_fontsize)
    _finalize_ax(ax, title=title, title_fontsize=15)

    legend_elems = [Line2D([0], [0], color="black", lw=3, label="active edge (colour = internal path pair)")]
    if show_labels and any_negative:
        legend_elems.append(
            Line2D(
                [0], [0], marker="d", color="w", markerfacecolor="white", markeredgecolor="black",
                markersize=9, label="negative weight",
            )
        )
    ax.legend(
        handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.05),
        ncol=1, fontsize=10, frameon=False,
    )

    fig.tight_layout()

    saved_paths: List[Path] = []
    if out_stub is not None:
        out_stub = Path(out_stub)
        out_stub.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            p = out_stub.with_suffix(f".{fmt}")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            saved_paths.append(p)

    if show:
        plt.show()
    else:
        plt.close(fig)

    rows = edge_table(edges, weights, weight_threshold=weight_threshold)
    return rows, saved_paths


def plot_pytheus_graph_split_by_mode(
    edges: Sequence[Tuple[int, int, int, int]],
    weights: Sequence[float],
    n: int,
    title: str = "",
    out_stub: Union[str, Path, None] = None,
    formats: Iterable[str] = ("png",),
    figsize: float = 12.0,
    curvature: float = 0.14,
    weight_threshold: float = 0.0,
    multi_edge_spacing: float = 0.6,
    linewidth_scale: float = 1.0,
    rotation: float = 0.0,
    clockwise: bool = False,
    show: bool = False,
):
    """
    Split-by-internal-mode panel view: one subplot per (c1, c2) colour pair,
    each showing only that edge family (with weight labels, since each panel
    has far fewer edges than the full dense graph).

    Returns (rows, saved_paths); `rows` is the full traceability table for
    all edges (as in `plot_pytheus_graph`).
    """
    edges = [tuple(int(v) for v in e) for e in edges]
    weights = [float(w) for w in weights]
    assert len(edges) == len(weights), "edges and weights must be the same length"

    weight_by_edge = dict(zip(edges, weights))
    color_pairs = sorted(set((e[2], e[3]) for e in edges))

    verts = vertex_positions(n, rotation=rotation, clockwise=clockwise)

    ncols = 2
    nrows = int(np.ceil(len(color_pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize, figsize))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for ax, (c1, c2) in zip(axes_flat, color_pairs):
        sub_edges = [e for e in edges if (e[2], e[3]) == (c1, c2)]
        sub_weights = [weight_by_edge[e] for e in sub_edges]
        active_sub = sum(1 for w in sub_weights if abs(w) > weight_threshold)

        _draw_edges(
            ax, sub_edges, sub_weights, verts,
            weight_threshold=weight_threshold, curvature=curvature,
            multi_edge_spacing=multi_edge_spacing, show_labels=True,
            uniform_style=False, label_fontsize=12, linewidth_scale=linewidth_scale,
        )
        _draw_nodes(ax, verts, node_radius=0.13, node_fontsize=15)
        _finalize_ax(ax, node_radius=0.13, title=f"path pair ({c1},{c2}) — {active_sub} active edge(s)", title_fontsize=13)

    for ax in axes_flat[len(color_pairs):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    saved_paths: List[Path] = []
    if out_stub is not None:
        out_stub = Path(out_stub)
        out_stub.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            p = out_stub.with_suffix(f".{fmt}")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            saved_paths.append(p)

    if show:
        plt.show()
    else:
        plt.close(fig)

    rows = edge_table(edges, weights, weight_threshold=weight_threshold)
    return rows, saved_paths
