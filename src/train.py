"""One-epoch verification training loop with TensorBoard logging."""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.dataset import BATCH_SIZE, load_split
from src.model import build_harvest_loss_predictor, trainable_parameter_count

LOG_DIR = "runs/harvestsafe_experiment_1"
CHECKPOINT_PATH = "runs/harvestsafe_model_epoch1.keras"
METRICS_PATH = "runs/harvestsafe_metrics.json"


def _weight_tag(weight) -> str:
    """Layer path, so histograms from different Dense kernels do not share a tag."""
    path = getattr(weight, "path", None) or weight.name
    return str(path).replace("/", ".")


def _dataset(features, target, shuffle: bool, seed: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((features, target))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=int(features.shape[0]),
            seed=seed,
            reshuffle_each_iteration=False,
        )
    return dataset.batch(BATCH_SIZE)


def _mean_metrics(model: tf.keras.Model, features, target) -> tuple[float, float]:
    """MSE and MAE with dropout off and batch-norm running statistics."""
    prediction = model(features, training=False)
    mse = tf.keras.losses.MeanSquaredError()(target, prediction)
    mae = tf.keras.losses.MeanAbsoluteError()(target, prediction)
    return float(mse), float(mae)


def run_training(
    train_path: str | Path,
    val_path: str | Path,
    log_dir: str | Path = LOG_DIR,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    metrics_path: str | Path = METRICS_PATH,
    epochs: int = 1,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train the Keras predictor and write a TensorBoard run plus checkpoint.

    The assignment verification rule fixes ``epochs`` at 1. Mean squared error
    is the optimised objective; mean absolute error is logged alongside it.
    """
    if epochs != 1:
        raise ValueError("Verification training is fixed at 1 epoch")

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    train = load_split(train_path)
    validation = load_split(val_path)
    train_dataset = _dataset(train["features"], train["target"], shuffle=True, seed=seed)
    model = build_harvest_loss_predictor(n_features=int(train["features"].shape[1]), seed=seed)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()

    log_dir = Path(log_dir)
    checkpoint_path = Path(checkpoint_path)
    metrics_path = Path(metrics_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.create_file_writer(str(log_dir))
    sample = tf.constant(train["features"][:BATCH_SIZE])

    @tf.function
    def _traced(features):
        return model(features, training=False)

    tf.summary.trace_on(graph=True)
    _ = _traced(sample)
    with writer.as_default():
        tf.summary.trace_export(name="HarvestLossPredictor", step=0)
    tf.summary.trace_off()

    global_step = 0
    train_mse = train_mae = val_mse = val_mae = float("nan")
    n_parameters = trainable_parameter_count(model)
    last_gradients: list[tf.Tensor] = []

    for epoch in range(epochs):
        running_mse = 0.0
        running_mae = 0.0
        seen = 0
        for features, target in train_dataset:
            with tf.GradientTape() as tape:
                prediction = model(features, training=True)
                loss = mse_loss(target, prediction)
            gradients = tape.gradient(loss, model.trainable_weights)
            last_gradients = gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            batch_mae = mae_loss(target, prediction)
            with writer.as_default():
                tf.summary.scalar("Loss/train_batch", loss, global_step)
                tf.summary.scalar("MAE/train_batch", batch_mae, global_step)
                tf.summary.scalar("GradNorm/total", tf.linalg.global_norm(gradients), global_step)

            size = int(features.shape[0])
            running_mse += float(loss) * size
            running_mae += float(batch_mae) * size
            seen += size
            global_step += 1

        train_mse = running_mse / seen
        train_mae = running_mae / seen
        val_mse, val_mae = _mean_metrics(model, validation["features"], validation["target"])

        with writer.as_default():
            tf.summary.scalar("Loss/train", train_mse, epoch)
            tf.summary.scalar("Loss/val", val_mse, epoch)
            tf.summary.scalar("MAE/train", train_mae, epoch)
            tf.summary.scalar("MAE/val", val_mae, epoch)
            for weight in model.trainable_weights:
                tf.summary.histogram(f"Weights/{_weight_tag(weight)}", weight, epoch)
            for weight, gradient in zip(model.trainable_weights, last_gradients):
                if gradient is not None:
                    tf.summary.histogram(f"Gradients/{_weight_tag(weight)}", gradient, epoch)

        print(
            f"  epoch {epoch + 1}/{epochs}  "
            f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
            f"train_mae={train_mae:.6f}  val_mae={val_mae:.6f}"
        )

    writer.flush()
    writer.close()
    model.save(checkpoint_path)

    metrics = {
        "epochs": epochs,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "n_parameters": n_parameters,
        "train_batches": global_step,
        "log_dir": str(log_dir),
        "checkpoint_path": str(checkpoint_path),
        "feature_names": [str(name) for name in train["feature_names"].tolist()],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
