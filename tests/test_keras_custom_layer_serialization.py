import os
# Force CPU to avoid CUDA driver/arch mismatches on local dev and CI
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tempfile
import numpy as np
import tensorflow as tf

from cerebros.persistence.keras_io import save_model_and_meta, load_model_safe
from cerebros.utils.random import set_global_seed

# Also disable GPUs at the TensorFlow level in case the env var is ignored
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


@tf.keras.utils.register_keras_serializable(package="cerebros")
class SquareLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        # No weights
        super().build(input_shape)

    def call(self, inputs):
        return tf.math.square(inputs)

    def get_config(self):
        return {}


def _build_model_with_custom(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = SquareLayer()(inputs)
    x = tf.keras.layers.Dense(4, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mse")
    return model


def _make_data(n: int = 128, d: int = 3):
    rng = np.random.default_rng(123)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X ** 2).sum(axis=1, keepdims=True) + 0.05 * rng.standard_normal((n, 1)).astype(np.float32)
    return X, y


def test_custom_layer_serialization_roundtrip_no_custom_objects():
    set_global_seed(11, deterministic=True)
    X, y = _make_data(128, 3)
    Xv, yv = X[:32], y[:32]

    model_a = _build_model_with_custom(X.shape[1])
    model_a.fit(X, y, epochs=2, batch_size=16, shuffle=False, verbose=0, validation_data=(Xv, yv))

    preds_before = model_a.predict(Xv, verbose=0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "custom.keras")
        save_model_and_meta(model_a, path, {"seed": 11})

        # Load with safe_mode and no custom_objects, relying on registration
        model_b = load_model_safe(path, safe_mode=True)
        preds_after = model_b.predict(Xv, verbose=0)
        np.testing.assert_allclose(preds_after, preds_before, rtol=0, atol=1e-6)

        # Resume training continues deterministically if optimizer state persisted
        model_b.fit(X, y, epochs=2, batch_size=16, shuffle=False, verbose=0, validation_data=(Xv, yv))

        set_global_seed(11, deterministic=True)
        model_c = _build_model_with_custom(X.shape[1])
        model_c.fit(X, y, epochs=4, batch_size=16, shuffle=False, verbose=0, validation_data=(Xv, yv))

        for w_loaded, w_full in zip(model_b.get_weights(), model_c.get_weights()):
            np.testing.assert_allclose(w_loaded, w_full, rtol=0, atol=1e-6)
