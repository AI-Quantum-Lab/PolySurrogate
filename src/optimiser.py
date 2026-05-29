"""
Clean surrogate inverse optimiser.

Main workflow:
1. Load configuration from optimiser_config.py
2. Load or generate initial graph samples
3. Load trained surrogate model
4. Set up PyTheus verification
5. Optimise graph weights toward a target state
6. Prune weak weights
7. Save plots, JSON summaries, and best graph

Helper functions are defined in:
- optimisation_utils.py
- target_states.py
"""

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from optimiser_config import OPTIMISER_CONFIG as CFG

from optimisation_utils import (
    build_dream_step,
    build_forward_metrics,
    build_optimizer,
    compute_grad_norm,
    fidelity_np,
    is_exact_zero_vector,
    load_trained_model,
    prepare_input_data,
    progressive_threshold_prune,
    pytheus_state_from_x,
    save_edge_fidelity_plot,
    save_gradient_plot,
    save_json,
    save_loss_fid_plot,
    save_time_plots,
    scalar_k_value,
    setup_pytheus_catalog,
    verify_with_pytheus_from_pred,
)
from target_states import get_target_state


# =============================================================================
# Main experiment
# =============================================================================


def run_optimisation(cfg=CFG):
    target_state_np = get_target_state(cfg["target_name"], cfg["n"])
    target_state_jax = jnp.asarray(target_state_np, dtype=jnp.float32)

    data = prepare_input_data(cfg)

    edges, tensor, mask = setup_pytheus_catalog(cfg["n"])
    tensor_jax = jnp.asarray(tensor, dtype=jnp.int32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)

    apply_fn, params = load_trained_model(cfg)

    optimizer, lr_schedule = build_optimizer(cfg)
    forward_metrics = build_forward_metrics(apply_fn, params, cfg)
    dream_step = build_dream_step(
        forward_metrics=forward_metrics,
        optimizer=optimizer,
        clip_min=cfg.get("clip_min", -1.0),
        clip_max=cfg.get("clip_max", 1.0),
    )

    results_root = Path(cfg["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    output_dir = results_root / cfg["folder_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "cfg.json", cfg)

    log_file = output_dir / "log.txt"
    log_fh = log_file.open("w", buffering=1)

    def write(*args, sep=" ", end="\n"):
        log_fh.write(sep.join(str(a) for a in args) + end)

    write("=== Inverse optimisation started ===")
    write(f"JAX devices: {jax.devices()}")
    write(f"Target: {cfg['target_name']}, n={cfg['n']}")
    write(f"Model path: {cfg['model_path']}")
    write(f"Output directory: {output_dir}")

    zero_state_samples = []
    best_solution = {
        "sample_id": None,
        "num_edges": np.inf,
        "graph_vector": None,
        "nn_state": None,
        "pytheus_state": None,
        "fidelity": None,
    }

    initial_fidelities = []
    final_fidelities = []
    final_pytheus_fidelities = []

    time_per_sample_list = []
    optimisation_time_list = []
    avg_time_per_step_list = []
    optimisation_steps_list = []

    edge_list = []
    py_fid_list = []

    store_step_vectors = bool(cfg.get("store_step_vectors", True))

    try:
        for k_value, dataset_X, dataset_Y in data:
            write(f"\n=== DATASET k={k_value} ===")

            for run_id, x0_np in enumerate(dataset_X):
                write(f"\n===== SAMPLE {run_id} =====")

                sample_start = time.perf_counter()

                x = jnp.asarray(x0_np, dtype=jnp.float32)
                opt_state = optimizer.init(x)

                loss_history = []
                fid_history = []
                mae_history = []
                l1_history = []
                update_norm_history = []
                grad_norm_history = []
                x_history = []
                lr_history = []

                grad_vector_history = []
                raw_update_history = []
                update_before_clip_history = []
                update_after_clip_history = []

                # Initial metrics
                step = 0
                init_loss, init_pred, init_fid, init_mae, init_l1 = forward_metrics(
                    x,
                    target_state_jax,
                )

                init_loss = jax.block_until_ready(init_loss)
                init_pred = jax.block_until_ready(init_pred)
                init_fid = jax.block_until_ready(init_fid)
                init_mae = jax.block_until_ready(init_mae)
                init_l1 = jax.block_until_ready(init_l1)

                nn_fid, py_fid, nn_state, py_state = verify_with_pytheus_from_pred(
                    x=x,
                    step=step,
                    loss=init_loss,
                    mae=init_mae,
                    l1=init_l1,
                    nn_state=init_pred,
                    nn_fid=init_fid,
                    y_target_np=target_state_np,
                    write_fn=write,
                    tensor_jax=tensor_jax,
                    mask_jax=mask_jax,
                )

                initial_fidelities.append(float(nn_fid))

                sample_info = {
                    "k_value": scalar_k_value(k_value),
                    "sample_id": int(run_id),
                    "x_init": np.asarray(x0_np, dtype=np.float32),
                    "y_target": target_state_np,
                    "init_loss": float(init_loss),
                    "init_mae": float(init_mae),
                    "init_l1": float(init_l1),
                    "nn_state_init": nn_state,
                    "py_state_init": py_state,
                    "nn_fid_init": float(nn_fid),
                    "py_fid_init": float(py_fid),
                    "sample_time": 0.0,
                    "optimisation_time": 0.0,
                    "avg_time_per_step": 0.0,
                    "optimisation_steps": 0,
                }

                last_x = np.asarray(x0_np, dtype=np.float32)
                optim_start = time.perf_counter()

                # Optimisation loop
                step = 0
                while step < cfg["num_steps"] and nn_fid < cfg["early_stop_nn_fid"]:
                    step += 1

                    current_lr = float(lr_schedule(step - 1))
                    lr_history.append(current_lr)

                    x_before = np.asarray(x, dtype=np.float32)

                    (
                        x,
                        opt_state,
                        loss_p,
                        y_pred,
                        nn_fid_jax,
                        mae_p,
                        l1_p,
                        grads_jax,
                        raw_update_jax,
                        update_before_clip_jax,
                        update_after_clip_jax,
                    ) = dream_step(x, opt_state, target_state_jax)

                    loss_p = jax.block_until_ready(loss_p)
                    y_pred = jax.block_until_ready(y_pred)
                    nn_fid_jax = jax.block_until_ready(nn_fid_jax)
                    mae_p = jax.block_until_ready(mae_p)
                    l1_p = jax.block_until_ready(l1_p)
                    x = jax.block_until_ready(x)
                    grads_jax = jax.block_until_ready(grads_jax)
                    raw_update_jax = jax.block_until_ready(raw_update_jax)
                    update_before_clip_jax = jax.block_until_ready(update_before_clip_jax)
                    update_after_clip_jax = jax.block_until_ready(update_after_clip_jax)

                    x_after = np.asarray(x, dtype=np.float32)
                    update_norm = float(np.linalg.norm(x_after - x_before))
                    grad_norm = compute_grad_norm(grads_jax)

                    grad_norm_history.append(grad_norm)
                    update_norm_history.append(update_norm)
                    loss_history.append(float(loss_p))
                    mae_history.append(float(mae_p))
                    l1_history.append(float(l1_p))
                    fid_history.append(float(nn_fid_jax))

                    if store_step_vectors:
                        grad_vector_history.append(np.asarray(grads_jax, dtype=np.float32).copy())
                        raw_update_history.append(np.asarray(raw_update_jax, dtype=np.float32).copy())
                        update_before_clip_history.append(
                            np.asarray(update_before_clip_jax, dtype=np.float32).copy()
                        )
                        update_after_clip_history.append(
                            np.asarray(update_after_clip_jax, dtype=np.float32).copy()
                        )

                    if step % cfg["verify_every"] == 0 or step == cfg["num_steps"]:
                        nn_fid, py_fid, nn_state, py_state = verify_with_pytheus_from_pred(
                            x=x,
                            step=step,
                            loss=loss_p,
                            mae=mae_p,
                            l1=l1_p,
                            nn_state=y_pred,
                            nn_fid=nn_fid_jax,
                            y_target_np=target_state_np,
                            write_fn=write,
                            tensor_jax=tensor_jax,
                            mask_jax=mask_jax,
                            grad_norm=grad_norm,
                            update_norm=update_norm,
                            print_lines=(step % cfg["print_every"] == 0),
                        )
                    else:
                        nn_fid = float(nn_fid_jax)

                    last_x = np.asarray(x, dtype=np.float32)
                    x_history.append(last_x.copy())

                # Pruning
                x_pruned, nn_pruned, _, pruning_info = progressive_threshold_prune(
                    x_best=last_x,
                    y_target_np=target_state_np,
                    write=write,
                    apply_fn=apply_fn,
                    params=params,
                    tensor_jax=tensor_jax,
                    mask_jax=mask_jax,
                    normalize_model_output=cfg.get("normalize_model_output", True),
                    fid_tol=cfg.get("prune_fid_tolerance", 1e-4),
                    thresholds=cfg.get(
                        "prune_thresholds",
                        [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    ),
                )

                x_history.append(np.asarray(x_pruned, dtype=np.float32).copy())

                jax.block_until_ready(jnp.asarray(x))
                optimisation_time = time.perf_counter() - optim_start
                avg_time_per_step = optimisation_time / max(step, 1)

                final_edges = int(np.sum(np.abs(x_pruned) > 1e-4))
                final_nn_state = np.asarray(nn_pruned, dtype=np.float32)
                final_py_state, _ = pytheus_state_from_x(x_pruned, tensor_jax, mask_jax)

                py_fid_pruned = fidelity_np(final_py_state, target_state_np)
                nn_fid_pruned = fidelity_np(final_nn_state, target_state_np)

                edge_list.append(final_edges)
                py_fid_list.append(float(py_fid_pruned))

                final_fidelities.append(float(nn_fid_pruned))
                final_pytheus_fidelities.append(float(py_fid_pruned))

                optimisation_time_list.append(float(optimisation_time))
                avg_time_per_step_list.append(float(avg_time_per_step))
                optimisation_steps_list.append(int(step))

                if final_edges < best_solution["num_edges"]:
                    best_solution.update(
                        {
                            "sample_id": int(run_id),
                            "num_edges": int(final_edges),
                            "graph_vector": np.asarray(x_pruned, dtype=np.float32).tolist(),
                            "nn_state": final_nn_state.tolist(),
                            "pytheus_state": np.asarray(final_py_state, dtype=np.float32).tolist(),
                            "fidelity": float(py_fid_pruned),
                        }
                    )
                    write("NEW BEST GRAPH FOUND")
                    write(f"sample = {run_id}")

                if is_exact_zero_vector(final_nn_state):
                    zero_entry = {
                        "sample_id": int(run_id),
                        "graph_vector": np.asarray(x_pruned, dtype=np.float32).tolist(),
                        "nn_state": final_nn_state.tolist(),
                        "pytheus_state": np.asarray(final_py_state, dtype=np.float32).tolist(),
                        "nn_norm": float(np.linalg.norm(final_nn_state)),
                        "py_norm": float(np.linalg.norm(final_py_state)),
                        "optimisation_steps": int(step),
                    }
                    zero_state_samples.append(zero_entry)
                    write("ZERO VECTOR FOUND")
                    write("sample:", run_id)
                    write("graph:", x_pruned)

                sample_dir = output_dir / f"sample_{run_id}"
                sample_dir.mkdir(exist_ok=True)

                abs_error = np.abs(final_nn_state - target_state_np)

                save_loss_fid_plot(
                    sample_dir=sample_dir,
                    loss_history=loss_history,
                    fid_history=fid_history,
                    target_name=cfg["target_name"],
                    nphotons=cfg["n"],
                )
                save_gradient_plot(sample_dir, grad_norm_history)

                sample_time = time.perf_counter() - sample_start
                time_per_sample_list.append(float(sample_time))

                sample_info.update(
                    {
                        "x_final": np.asarray(x_pruned, dtype=np.float32),
                        "nn_state_final": final_nn_state,
                        "py_state_final": final_py_state,
                        "loss_history": loss_history,
                        "fid_history": fid_history,
                        "mae_history": mae_history,
                        "l1_history": l1_history,
                        "grad_norm_history": grad_norm_history,
                        "update_norm_history": update_norm_history,
                        "pruning_log": pruning_info,
                        "abs_error": abs_error,
                        "final_edges": int(final_edges),
                        "nn_fid_final": float(nn_fid_pruned),
                        "py_fid_final": float(py_fid_pruned),
                        "sample_time": float(sample_time),
                        "optimisation_time": float(optimisation_time),
                        "avg_time_per_step": float(avg_time_per_step),
                        "optimisation_steps": int(step),
                        "x_history": x_history,
                        "lr_history": lr_history,
                        "grad_vector_history": grad_vector_history,
                        "raw_update_history": raw_update_history,
                        "update_before_clip_history": update_before_clip_history,
                        "update_after_clip_history": update_after_clip_history,
                    }
                )

                save_json(sample_dir / "sample_info_file.json", sample_info)

        save_edge_fidelity_plot(output_dir, edge_list, py_fid_list, cfg)
        save_time_plots(output_dir, time_per_sample_list, cfg["n"])

        save_json(output_dir / "zero_state_samples.json", zero_state_samples)

        summary_data = {
            "initial_fidelities": [float(x) for x in initial_fidelities],
            "final_fidelities": [float(x) for x in final_fidelities],
            "final_pytheus_fidelities": [float(x) for x in final_pytheus_fidelities],
            "time_per_sample": [float(x) for x in time_per_sample_list],
            "optimisation_time_per_sample": [float(x) for x in optimisation_time_list],
            "avg_time_per_step": [float(x) for x in avg_time_per_step_list],
            "optimisation_steps": [int(x) for x in optimisation_steps_list],
            "final_edges_per_sample": [int(x) for x in edge_list],
        }

        save_json(output_dir / "optimisation_summary.json", summary_data)
        save_json(output_dir / "best_graph_solution.json", best_solution)

    finally:
        log_fh.close()

    return output_dir


if __name__ == "__main__":
    output_dir = run_optimisation(CFG)
    print(f"Optimisation complete. Results saved in: {output_dir}")
