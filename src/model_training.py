"""
Clean surrogate model training script.

Main workflow:
1. Load configuration from config.py
2. Load generated dataset
3. Build FNN/PNN model
4. Train using MAE / normalized MAE loss
5. Save checkpoints, plots, logs, parameters, and test metrics
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, checkpoints
from flax.serialization import to_bytes

from training_config import TRAINING_CONFIG as CFG

from models import create_model
from model_training_utils import (
    create_run_dirs,
    log_file,
    write_json,
    save_training_plot,
    save_test_plot,
    save_params_msgpack,
    load_and_split_dataset,
    make_epoch_perm,
    create_train_step,
    create_train_epoch,
    create_eval_step,
    create_test_step,
    eval_epoch,
    test_epoch,
    create_cosine_adamw_optimizer,
    get_lr_for_epoch,
)


# =============================================================================
# Configuration
# =============================================================================

NODES = CFG["NODES"]
DIMENSIONS = CFG["DIMENSIONS"]
DATE = CFG["DATE"]

MODEL_NAME = CFG["MODEL_NAME"]
HIDDEN_DIM = CFG["HIDDEN_DIM"]

DATA_PATH = CFG["DATA_PATH"]
DATA_SIZE = CFG["DATA_SIZE"]

NORMALIZE_MODEL_OUTPUT = CFG["NORMALIZE_MODEL_OUTPUT"]

TRAIN_SPLIT = CFG["TRAIN_SPLIT"]
VAL_SPLIT = CFG["VAL_SPLIT"]
TEST_SPLIT = CFG["TEST_SPLIT"]

LEARNING_RATE = CFG["LEARNING_RATE"]
LR_AFTER_DECAY = CFG["LR_AFTER_DECAY"]
LR_DECAY_UNTIL_EPOCH = CFG["LR_DECAY_UNTIL_EPOCH"]

BATCH_SIZE = CFG["BATCH_SIZE"]
NUM_EPOCHS = CFG["NUM_EPOCHS"]
PATIENCE = CFG["PATIENCE"]
TOLERANCE = CFG["TOLERANCE"]
INIT_KEY = CFG["INIT_KEY"]

RESUME_FULL_STATE = CFG["RESUME_FULL_STATE"]
CKPT_DIR_RESTORE = CFG["CKPT_DIR_RESTORE"]

ROOT_FOLDER = CFG["ROOT_FOLDER"]
RUN_NAME = CFG["RUN_NAME"]
LOSS_NAME = CFG["LOSS_NAME"]

INPUT_DIMS = 2 * NODES * (NODES - 1)
OUTPUT_DIMS = 2 ** NODES

PLOT_TITLE = f"Training loss, {MODEL_NAME} (n={NODES}, hidden={HIDDEN_DIM})"


# =============================================================================
# Main training function
# =============================================================================

def train_surrogate_model():
    model_info = {
        "description": f"{MODEL_NAME}, {NODES}-node case",
        "root_folder": ROOT_FOLDER,
        "Folder": RUN_NAME,
        "Nodes": NODES,
        "Dimensions": DIMENSIONS,
        "model_name": MODEL_NAME,
        "architecture": HIDDEN_DIM,
        "input_dims": INPUT_DIMS,
        "output_dims": OUTPUT_DIMS,
        "batch_size": BATCH_SIZE,
        "num_epoch": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "lr_schedule": {
            "type": "cosine_decay",
            "initial_lr": LEARNING_RATE,
            "final_lr": LR_AFTER_DECAY,
            "decay_until_epoch": LR_DECAY_UNTIL_EPOCH,
        },
        "init_key": INIT_KEY,
        "patience": PATIENCE,
        "tolerance": TOLERANCE,
        "train_split": TRAIN_SPLIT,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "dataset": DATA_PATH,
        "data_size": DATA_SIZE,
        "normalize_model_output": NORMALIZE_MODEL_OUTPUT,
        "train_cont_full_state": RESUME_FULL_STATE,
        "CKPT_DIR_restore": CKPT_DIR_RESTORE,
        "loss_name": LOSS_NAME,
        "plot_title": PLOT_TITLE,
    }

    # -------------------------------------------------------------------------
    # Create output folders and logs
    # -------------------------------------------------------------------------
    run_dir, ckpt_dir, plot_dir, snap_dir, log_file_path = create_run_dirs(
        ROOT_FOLDER,
        RUN_NAME,
    )

    log_file(log_file_path, f"Run directory: {run_dir}")
    log_file(log_file_path, f"JAX devices: {jax.devices()}")
    log_file(log_file_path, f"Default backend: {jax.default_backend()}")

    model_info_path = run_dir / "model_info.json"
    write_json(model_info_path, model_info)

    # -------------------------------------------------------------------------
    # Build model, optimizer, train state
    # -------------------------------------------------------------------------
    model = create_model(
        model_name=MODEL_NAME,
        hidden_dims=HIDDEN_DIM,
        out_dims=OUTPUT_DIMS,
        nodes=NODES,
    )

    key = jax.random.PRNGKey(INIT_KEY)
    key, init_key = jax.random.split(key)

    x_example = jnp.zeros((1, INPUT_DIMS), dtype=jnp.float32)
    params = model.init(init_key, x_example)

    effective_data_size = DATA_SIZE
    if effective_data_size is None:
        # load_and_split_dataset will use all samples, but the scheduler needs
        # a number. We set it after loading if DATA_SIZE is None.
        effective_data_size = 1

    steps_per_epoch = int((effective_data_size * TRAIN_SPLIT) // BATCH_SIZE)
    steps_per_epoch = max(steps_per_epoch, 1)

    tx = create_cosine_adamw_optimizer(
        learning_rate=LEARNING_RATE,
        lr_after_decay=LR_AFTER_DECAY,
        decay_until_epoch=LR_DECAY_UNTIL_EPOCH,
        steps_per_epoch=steps_per_epoch,
        weight_decay=1e-4,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # -------------------------------------------------------------------------
    # Optional resume full state
    # -------------------------------------------------------------------------
    if RESUME_FULL_STATE:
        ckpt_target = {
            "state": state,
            "epoch": -1,
            "best_val": np.inf,
            "best_epoch": -1,
            "wait": 0,
        }

        ckpt = checkpoints.restore_checkpoint(
            CKPT_DIR_RESTORE,
            target=ckpt_target,
        )

        state = ckpt["state"]
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        best_epoch = int(ckpt["best_epoch"])
        wait = int(ckpt["wait"])
        log_file(log_file_path, f"Resuming from epoch: {start_epoch}")
    else:
        start_epoch = 0
        best_val = np.inf
        best_epoch = -1
        wait = 0

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data_loading_start = time.time()

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_split_dataset(
        dataset_path=DATA_PATH,
        n_samples=DATA_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
    )

    X_train = jax.device_put(jnp.asarray(X_train, dtype=jnp.float32))
    Y_train = jax.device_put(jnp.asarray(Y_train, dtype=jnp.float32))
    X_val = jax.device_put(jnp.asarray(X_val, dtype=jnp.float32))
    Y_val = jax.device_put(jnp.asarray(Y_val, dtype=jnp.float32))
    X_test = jax.device_put(jnp.asarray(X_test, dtype=jnp.float32))
    Y_test = jax.device_put(jnp.asarray(Y_test, dtype=jnp.float32))

    data_loading_time = time.time() - data_loading_start
    log_file(log_file_path, f"Time for loading + splitting data: {data_loading_time}")

    # If DATA_SIZE was None, update scheduler-related metadata.
    model_info["actual_train_size"] = int(X_train.shape[0])
    model_info["actual_val_size"] = int(X_val.shape[0])
    model_info["actual_test_size"] = int(X_test.shape[0])
    write_json(model_info_path, model_info)

    # -------------------------------------------------------------------------
    # Create JIT functions
    # -------------------------------------------------------------------------
    train_step = create_train_step(normalize_output=NORMALIZE_MODEL_OUTPUT)
    train_epoch = create_train_epoch(train_step)

    eval_step = create_eval_step(
        apply_fn=state.apply_fn,
        normalize_output=NORMALIZE_MODEL_OUTPUT,
    )

    test_step = create_test_step(
        apply_fn=state.apply_fn,
        normalize_output=NORMALIZE_MODEL_OUTPUT,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    train_losses = []
    val_losses = []
    epochs = []

    early_stop = False
    best_state = state

    train_key = key
    training_time_computation = 0.0
    training_time_computation_list = []
    epoch_time_list = []

    full_loop_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_compute_start = time.time()

        train_key, epoch_key = jax.random.split(train_key)
        perm_batches = make_epoch_perm(X_train.shape[0], BATCH_SIZE, epoch_key)

        epoch_train_start = time.time()
        state, train_loss = train_epoch(state, X_train, Y_train, perm_batches)
        epoch_train_time = time.time() - epoch_train_start
        epoch_time_list.append(epoch_train_time)

        val_loss = eval_epoch(
            state.params,
            X_val,
            Y_val,
            eval_step,
            batch_size=BATCH_SIZE,
        )

        epoch_compute_time = time.time() - epoch_compute_start
        training_time_computation += epoch_compute_time
        training_time_computation_list.append(training_time_computation)

        num_batch = perm_batches.shape[0]
        avg_time_batch = float(epoch_train_time) / float(num_batch)

        current_lr = get_lr_for_epoch(
            epoch,
            learning_rate=LEARNING_RATE,
            lr_after_decay=LR_AFTER_DECAY,
            decay_until_epoch=LR_DECAY_UNTIL_EPOCH,
        )

        # Checkpoint on validation improvement
        if val_loss < best_val - TOLERANCE:
            best_val = float(val_loss)
            best_state = state
            best_epoch = epoch
            wait = 0

            snap_path = snap_dir / f"best_param_epoch_{epoch}.msgpack"
            with snap_path.open("wb") as f:
                f.write(to_bytes(best_state.params))

            checkpoints.save_checkpoint(
                ckpt_dir,
                {
                    "state": state,
                    "epoch": epoch,
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "wait": int(wait),
                },
                step=epoch,
                keep=5,
            )

            log_file(log_file_path, f"New checkpoint saved at Epoch {epoch}")

        else:
            wait += 1
            if wait >= PATIENCE:
                log_file(log_file_path, f"EARLY STOP at epoch {epoch}, best at {best_epoch}")
                early_stop = True
                break

        log_file(
            log_file_path,
            f"Epoch={epoch:5d} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={float(train_loss):.8f} | "
            f"val_loss={float(val_loss):.6f} | "
            f"epoch_compute_time={epoch_compute_time:.6f} | "
            f"train_time_per_epoch={epoch_train_time:.6f} | "
            f"wait={wait} | "
            f"patience={PATIENCE} | "
            f"avg_batch_time={avg_time_batch:.6f} | "
        )

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        epochs.append(epoch)

        if epoch % 100 == 0:
            save_training_plot(
                epochs,
                train_losses,
                val_losses,
                plot_dir / "training_curves.png",
                LOSS_NAME,
                PLOT_TITLE,
            )

        model_info["train_loss"] = train_losses
        model_info["val_loss"] = val_losses
        model_info["best_val_loss"] = best_val
        model_info["best_epoch"] = best_epoch
        model_info["data_loading_time"] = data_loading_time
        model_info["cumulative_compute_time"] = training_time_computation_list
        model_info["total_compute_time"] = training_time_computation
        model_info["train_step_time_list"] = epoch_time_list
        write_json(model_info_path, model_info)

    full_loop_time = time.time() - full_loop_start

    log_file(log_file_path, f"Total compute time: {training_time_computation / 60:.6f} mins")
    log_file(log_file_path, f"Total compute time: {training_time_computation / 3600:.6f} hours")
    log_file(log_file_path, f"Total wall time: {full_loop_time / 60:.6f} mins")
    log_file(log_file_path, f"Total wall time: {full_loop_time / 3600:.6f} hours")

    # -------------------------------------------------------------------------
    # Restore best model and save params
    # -------------------------------------------------------------------------
    if best_state is not None:
        state = best_state
        log_file(log_file_path, f"Restored best params from epoch {best_epoch}")
    else:
        log_file(log_file_path, "No best state saved, using last state")

    params_path = run_dir / "params.msgpack"
    save_params_msgpack(state.params, params_path)
    log_file(log_file_path, f"Saved params to {params_path}")

    # -------------------------------------------------------------------------
    # Test best model
    # -------------------------------------------------------------------------
    (
        loss_mean,
        fidelity_mean,
        mse_mean,
        mae_mean,
        loss_list,
        fid_list,
        mse_list,
        mae_list,
    ) = test_epoch(
        state.params,
        X_test,
        Y_test,
        state.apply_fn,
        test_step,
        batch_size=20,
    )

    log_file(
        log_file_path,
        f"Test Loss: {loss_mean} | fidelity: {fidelity_mean} | "
        f"mae: {mae_mean} | mse: {mse_mean}",
    )

    save_test_plot(
        fid_list,
        plot_dir / "test_fidelity_curve.png",
        title=f"Test Fidelity (n={NODES}, {LOSS_NAME})",
    )

    save_training_plot(
        epochs,
        train_losses,
        val_losses,
        plot_dir / "training_curves.png",
        LOSS_NAME,
        PLOT_TITLE,
    )

    np.savez(
        run_dir / "test_metrics.npz",
        loss=loss_list,
        fidelity=fid_list,
        mse=mse_list,
        mae=mae_list,
    )

    model_info["early_stop"] = early_stop
    model_info["train_loss"] = train_losses
    model_info["val_loss"] = val_losses
    model_info["best_loss"] = best_val
    model_info["best_epoch"] = best_epoch
    model_info["data_loading_time"] = data_loading_time
    model_info["cumulative_compute_time"] = training_time_computation_list
    model_info["total_compute_time"] = training_time_computation
    model_info["total_wall_time"] = full_loop_time
    model_info["train_step_time_list"] = epoch_time_list
    model_info["test_loss"] = float(loss_mean)
    model_info["test_fidelity"] = float(fidelity_mean)
    model_info["test_mse"] = float(mse_mean)
    model_info["test_mae"] = float(mae_mean)

    write_json(model_info_path, model_info)
    log_file(log_file_path, "Reached end of script!")

    return run_dir


if __name__ == "__main__":
    run_dir = train_surrogate_model()
    print(f"Training complete. Results saved in: {run_dir}")
