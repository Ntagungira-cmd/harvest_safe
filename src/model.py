"""HarvestSafe multi-layer perceptron for district post-harvest loss."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

def _kernel_initializer(seed: int) -> tf.keras.initializers.Initializer:
    """He/Kaiming normal for LeakyReLU(0.1): std = sqrt(2 / ((1 + a^2) * fan_in))."""
    return tf.keras.initializers.VarianceScaling(
        scale=2.0 / (1.0 + 0.1**2),
        mode="fan_in",
        distribution="untruncated_normal",
        seed=seed,
    )


def build_harvest_loss_predictor(n_features: int = 7, seed: int = 42) -> tf.keras.Model:
    """7 -> 32 -> 16 -> 1 network with a loss fraction bounded in [0, 0.50].

    LeakyReLU keeps a gradient on negative pre-activations so moisture
    features that fall below the batch mean do not permanently silence units.
    Dropout and batch normalization regularise a model that will later be
    trained on a small district panel. The sigmoid head is scaled by 0.50,
    the maximum theoretical loss fraction used in this study.
    """
    features = tf.keras.Input(shape=(n_features,), name="features")
    hidden = tf.keras.layers.Dense(
        32,
        kernel_initializer=_kernel_initializer(seed),
        bias_initializer="zeros",
        name="hidden_32",
    )(features)
    hidden = tf.keras.layers.BatchNormalization(name="hidden_32_norm")(hidden)
    hidden = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="hidden_32_activation")(hidden)
    hidden = tf.keras.layers.Dropout(0.2, seed=seed, name="hidden_32_dropout")(hidden)
    hidden = tf.keras.layers.Dense(
        16,
        kernel_initializer=_kernel_initializer(seed + 1),
        bias_initializer="zeros",
        name="hidden_16",
    )(hidden)
    hidden = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="hidden_16_activation")(hidden)
    hidden = tf.keras.layers.Dropout(0.1, seed=seed + 1, name="hidden_16_dropout")(hidden)
    logits = tf.keras.layers.Dense(
        1,
        kernel_initializer=_kernel_initializer(seed + 2),
        bias_initializer="zeros",
        name="loss_logit",
    )(hidden)
    loss_fraction = tf.keras.layers.Lambda(lambda logit: 0.50 * tf.nn.sigmoid(logit), name="loss_fraction")(
        logits
    )
    return tf.keras.Model(features, loss_fraction, name="HarvestLossPredictor")


def trainable_parameter_count(model: tf.keras.Model) -> int:
    """Count affine weights only. Batch-norm moving statistics are excluded."""
    return int(sum(tf.size(weight).numpy() for weight in model.trainable_weights))
