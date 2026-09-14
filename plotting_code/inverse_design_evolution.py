"""Plot the loss and graph evolution of one recorded GHZ n=4 design run.

The figure combines the full loss curve with graph snapshots at steps 0,
50, 100, and after final pruning. Only NumPy and Matplotlib are required.

Run from the repository root with:
    python plotting_code/inverse_design_evolution.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_INFO_PATH = SCRIPT_DIR / "Data" / "inverse_design_evolution" / "sample_info_file.json"

OUTPUT_DIR = SCRIPT_DIR / "Results" / "inverse_design_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_STEPS = [0, 50, 100, "final_pruned"]
STEP_TITLES = {0: "Random init (step 0)", 50: "Step 50", 100: "Step 100",
               "final_pruned": "Final (pruned)"}

# Fixed line width for every active edge; opacity represents |weight|.
EDGE_LINEWIDTH = 2.6
EDGE_ALPHA_MIN = 0.15
EDGE_ALPHA_MAX = 1.0

# Node-number label size inside each graph snapshot.
NODE_LABEL_FONTSIZE = 18.0

FIGSIZE = (13, 8.5)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ARROW_COLOR = "#E76F51"
CURVE_COLOR = "#2E86AB"
EDGE_COLORS = ["dodgerblue", "firebrick", "limegreen", "darkorange", "purple", "yellow", "cyan"]


def build_edge_catalog(vertices=4, dimensions=2):
    """Build the ordered, loop-free edge catalogue used by this figure."""
    return [
        (vertex_1, vertex_2, mode_1, mode_2)
        for vertex_1 in range(vertices)
        for vertex_2 in range(vertex_1 + 1, vertices)
        for mode_1 in range(dimensions)
        for mode_2 in range(dimensions)
    ]


def group_edges_by_vertices(edges):
    """Group the internal-mode pairs belonging to each vertex pair."""
    grouped = {}
    for vertex_1, vertex_2, mode_1, mode_2 in edges:
        grouped.setdefault((vertex_1, vertex_2), []).append((mode_1, mode_2))
    return grouped


def vertex_positions(n, rotation=0.0, clockwise=False):
    """Return evenly spaced circular positions for numbered graph vertices."""
    angles = np.linspace(0, 2 * np.pi * (n - 1) / n, n)
    if clockwise:
        angles = -angles
    angles += rotation
    radius = 0.9
    return {i: (radius * np.cos(angle), radius * np.sin(angle)) for i, angle in enumerate(angles)}


def quadratic_bezier(p0, p1, p2, t):
    """Evaluate a quadratic Bezier curve at coordinates ``t``."""
    t = t[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def draw_nodes(ax, vertices, radius=0.15, fontsize=19.0):
    """Draw graph vertices and their integer labels."""
    for index, (x_coord, y_coord) in vertices.items():
        ax.add_patch(
            plt.Circle(
                (x_coord, y_coord),
                radius,
                facecolor="lightgrey",
                edgecolor="dimgray",
                linewidth=2.2,
                zorder=10,
            )
        )
        ax.text(
            x_coord,
            y_coord,
            str(index),
            fontsize=fontsize,
            ha="center",
            va="center",
            zorder=11,
            fontweight="bold",
        )


def finalize_graph_axis(ax, node_radius=0.15):
    """Apply the common limits and clean styling to a graph axis."""
    limit = 1.0 + node_radius + 0.25
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_edges_opacity_only(ax, edges, weights, verts, curvature, multi_edge_spacing=0.75):
    """Draw active curved edges with fixed width and weight-based opacity.

    Inactive (zero-weight) edges are skipped entirely.
    """
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    grouped_edges = group_edges_by_vertices(edges)

    curve_t_half1 = np.linspace(0.0, 0.5, 24)
    curve_t_half2 = np.linspace(0.5, 1.0, 24)

    for (v1, v2), colorings in grouped_edges.items():
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
            if abs(w) <= 0.0:
                continue

            offset = (2 * ind - mult + 1) * multi_edge_spacing * curvature
            ctrl = mid + offset * rect

            pts1 = quadratic_bezier(vert1, ctrl, vert2, curve_t_half1)
            pts2 = quadratic_bezier(vert1, ctrl, vert2, curve_t_half2)

            col1 = EDGE_COLORS[int(c1) % len(EDGE_COLORS)]
            col2 = EDGE_COLORS[int(c2) % len(EDGE_COLORS)]

            alpha = EDGE_ALPHA_MIN + (EDGE_ALPHA_MAX - EDGE_ALPHA_MIN) * min(abs(w), 1.0)

            ax.plot(pts1[:, 0], pts1[:, 1], color=col1, linewidth=EDGE_LINEWIDTH,
                     alpha=alpha, solid_capstyle="round", zorder=5)
            ax.plot(pts2[:, 0], pts2[:, 1], color=col2, linewidth=EDGE_LINEWIDTH,
                     alpha=alpha, solid_capstyle="round", zorder=5)


def load_records():
    """Load the recorded loss, fidelity, and graph-weight trajectory.

    ``x_history[0]`` is the initial graph. Each later entry is the graph
    after that optimisation step. The pruned final graph is stored separately.
    """
    with open(SAMPLE_INFO_PATH, "r") as f:
        info = json.load(f)

    edge_tuples = build_edge_catalog(vertices=4, dimensions=2)

    n_steps = int(info["optimisation_steps"])
    x_history = info["x_history"]
    loss_history = info["loss_history"]
    fid_history = info["fid_history"]
    assert len(x_history) == n_steps + 1
    assert len(loss_history) == n_steps
    assert len(fid_history) == n_steps

    all_records = [
        {"step": 0, "loss": info["init_loss"], "fidelity": info["nn_fid_init"],
         "is_final_pruned": False},
    ]
    for step in range(1, n_steps + 1):
        all_records.append({
            "step": step,
            "loss": loss_history[step - 1],
            "fidelity": fid_history[step - 1],
            "is_final_pruned": False,
        })
    all_records.append({
        "step": "final_pruned",
        "loss": loss_history[-1],
        "fidelity": info["nn_fid_final"],
        "is_final_pruned": True,
    })

    W = np.asarray(x_history + [info["x_final"]], dtype=float)

    records = {r["step"]: r for r in all_records}
    return records, edge_tuples, W, all_records


def render_standalone_snapshot(step, record, edge_tuples, weights):
    """Save a standalone clean-style snapshot PNG (bonus/reference asset,
    not used inside the composite figure -- see draw_graph_on_ax for that).
    Same opacity-only edge encoding as the composite figure (fixed
    linewidth, alpha varies with |weight|), no per-edge numeric labels.
    """
    fid = record["fidelity"]
    title = STEP_TITLES[step]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw_graph_on_ax(ax, edge_tuples, weights)
    ax.set_title(f"{title} (fidelity={fid:.4f})", fontsize=15, fontweight="bold")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"snapshot_step_{step}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_graph_on_ax(ax, edge_tuples, weights):
    """Draw one graph snapshot directly into a given composite-figure axis:
    clean style (no per-edge numeric labels, fixed linewidth, alpha-only
    encodes |weight|), no legend, no title -- keeps the composite figure
    free of per-panel clutter. Caption is added separately by the caller.
    """
    verts = vertex_positions(4)
    weights_list = np.asarray(weights).tolist()
    active_count = int(np.sum(np.abs(weights) > 0.0))
    curvature = float(np.clip(0.11 + 0.006 * active_count, 0.11, 0.30))

    draw_edges_opacity_only(ax, edge_tuples, weights_list, verts, curvature=curvature)
    draw_nodes(ax, verts, fontsize=NODE_LABEL_FONTSIZE)
    finalize_graph_axis(ax)


def build_full_loss_curve(all_records):
    """Return the full trajectory with final pruning as one extra point."""
    pre_pruning = [r for r in all_records if not r["is_final_pruned"]]
    steps = np.array([r["step"] for r in pre_pruning])
    losses = np.array([r["loss"] for r in pre_pruning])

    final_rec = next(r for r in all_records if r["is_final_pruned"])
    final_step = steps[-1] + 1
    final_loss = losses[-1]  # pruning doesn't add a new recorded loss value

    return steps, losses, final_step, final_loss, final_rec


def make_figure(yscale, records, edge_tuples, W, all_records):
    steps, losses, final_step, final_loss, final_rec = build_full_loss_curve(all_records)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.15)

    ax_loss = fig.add_subplot(gs[0, :])

    ax_loss.plot(steps, losses, color=CURVE_COLOR, linewidth=2.2)
    ax_loss.scatter([final_step], [final_loss], color=ARROW_COLOR, marker="*",
                    s=220, zorder=5, edgecolor="black", linewidth=0.6)

    # NOTE on vertical spacing: the title is attached to ax_loss and is
    # positioned relative to the axes' own top edge (via `pad`, in points).
    # Shifting the whole axes down (e.g. via ax_loss.set_position) moves the
    # title along with it by the same amount, so it does NOT change the gap
    # between the title and the curve/top annotation -- that gap is set by
    # (a) the title's `pad` and (b) how much headroom the y-axis has above
    # the highest data point. Both are increased below to give real,
    # visible separation.
    max_loss = float(losses.max())
    if yscale == "log":
        ax_loss.set_yscale("log")
        y_top = 10 ** (np.ceil(np.log10(max_loss)) + 0.6)  # ~1 extra decade of headroom
    else:
        y_top = max_loss * 1.45  # ~45% headroom above the highest point

    ax_loss.set_xlabel("Optimisation step", fontweight="bold")
    ax_loss.set_ylabel("Loss", fontweight="bold")
    ax_loss.set_title(
        "GHZ (n = 4) inverse-design optimisation: loss curve and graph evolution (sample 0)",
        fontsize=19, fontweight="bold", pad=22,
    )
    ax_loss.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    for spine in ("top", "right"):
        ax_loss.spines[spine].set_visible(False)

    if yscale == "log":
        min_loss = float(losses.min())
        y_bottom = 10 ** (np.floor(np.log10(min_loss)) - 0.3)
    else:
        y_bottom = 0.0
    ax_loss.set_ylim(y_bottom, y_top)

    x_span = final_step - steps[0]
    ax_loss.set_xlim(steps[0] - 0.03 * x_span, final_step + 0.03 * x_span)

    # ---- marker + simple fidelity annotation at each selected step ----
    marker_points = []
    for step in SELECTED_STEPS:
        rec = records[step]
        if step == "final_pruned":
            x_pt, y_pt = final_step, final_loss
        else:
            x_pt, y_pt = step, rec["loss"]
        marker_points.append((x_pt, y_pt))
        ax_loss.scatter([x_pt], [y_pt], color=ARROW_COLOR, s=60, zorder=4,
                        edgecolor="black", linewidth=0.5)

        # The leftmost point (step 0) sits right next to the y-axis tick
        # labels -- anchor its annotation to the right instead of centered
        # so it doesn't overlap them.
        if step == 0:
            ha, xytext = "left", (8, 8)
        else:
            ha, xytext = "center", (0, 10)

        ax_loss.annotate(
            f"fid={rec['fidelity']:.3f}",
            xy=(x_pt, y_pt), xytext=xytext, textcoords="offset points",
            ha=ha, fontsize=14, fontweight="bold",
        )

    # ---- graph snapshot panels + connecting arrows ----
    for col, step in enumerate(SELECTED_STEPS):
        ax_img = fig.add_subplot(gs[1, col])
        row_idx = next(i for i, r in enumerate(all_records) if r["step"] == step)
        draw_graph_on_ax(ax_img, edge_tuples, W[row_idx])
        ax_img.text(0.5, -0.06, STEP_TITLES[step], transform=ax_img.transAxes,
                    ha="center", va="top", fontsize=16, fontweight="bold")

        x_pt, y_pt = marker_points[col]
        con = fig.add_artist(
            plt.matplotlib.patches.ConnectionPatch(
                xyA=(x_pt, y_pt), coordsA=ax_loss.transData,
                xyB=(0.5, 1.0), coordsB=ax_img.transAxes,
                color=ARROW_COLOR, linewidth=1.4,
                arrowstyle="-|>", mutation_scale=14,
            )
        )
        con.set_zorder(1)

    fig.text(
        0.5, 0.005,
        "Edge opacity = relative |weight| magnitude (line width is fixed).  "
        "Edge colour = internal path-pair identity.",
        ha="center", va="bottom", fontsize=13.5, fontweight="bold", color="0.15",
    )

    return fig


def main():
    records, edge_tuples, W, all_records = load_records()

    print("Saving standalone snapshot PNGs (step 0, 50, 100, final_pruned)...")
    for step in SELECTED_STEPS:
        rec = records[step]
        row_idx = next(i for i, r in enumerate(all_records) if r["step"] == step)
        weights = W[row_idx]
        path = render_standalone_snapshot(step, rec, edge_tuples, weights)
        print(f"  step={step}: {path.name}")

    for yscale in ["linear", "log"]:
        fig = make_figure(yscale, records, edge_tuples, W, all_records)
        basename = f"loss_evolution_ghz4_sample0_{yscale}"
        png_path = OUTPUT_DIR / f"{basename}.png"
        pdf_path = OUTPUT_DIR / f"{basename}.pdf"
        fig.savefig(png_path, dpi=250, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {png_path.name} / {pdf_path.name}")


if __name__ == "__main__":
    main()
