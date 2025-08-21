import os
# Force CPU to avoid CUDA driver/arch mismatches on local dev and CI
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# Set deterministic flags before importing TensorFlow
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

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


def _build_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def _make_data(n: int = 256, d: int = 6):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d, 1)).astype(np.float32)
    y = X @ w + 0.1 * rng.standard_normal((n, 1)).astype(np.float32)
    return X, y


def test_keras_persistence_roundtrip():
    # Deterministic seeds for reproducibility
    set_global_seed(7, deterministic=True)

    X, y = _make_data(256, 6)
    X_val, y_val = X[:64], y[:64]

    # Train a small model for a couple of epochs
    model_a = _build_model(X.shape[1])
    _ = model_a.fit(
        X,
        y,
        batch_size=32,
        epochs=2,
        shuffle=False,
        validation_data=(X_val, y_val),
        verbose=0,
    )

    preds_before = model_a.predict(X_val, verbose=0)

    with tempfile.TemporaryDirectory() as tmpd:
        save_path = os.path.join(tmpd, "model.keras")
        meta = {"seed": 7, "epochs_trained": 2}
        # Save model and meta
        save_model_and_meta(model_a, save_path, meta)

        # Load with safe_mode
        model_b = load_model_safe(save_path, safe_mode=True)
        assert model_b.optimizer is not None, "Optimizer state was not restored"

        preds_after = model_b.predict(X_val, verbose=0)

        # 1) Round-trip should not change predictions
        np.testing.assert_allclose(preds_after, preds_before, rtol=0, atol=1e-7)

        # 2) Resuming training should match uninterrupted training if optimizer state is restored
        _ = model_b.fit(
            X,
            y,
            batch_size=32,
            epochs=2,
            shuffle=False,
            validation_data=(X_val, y_val),
            verbose=0,
        )

        # Build a fresh model with the same seed and train 4 epochs uninterrupted
        set_global_seed(7, deterministic=True)
        model_c = _build_model(X.shape[1])
        _ = model_c.fit(
            X,
            y,
            batch_size=32,
            epochs=4,
            shuffle=False,
            validation_data=(X_val, y_val),
            verbose=0,
        )

        # 3) Final weights should be identical (within tight tolerance)
        for w_loaded, w_full in zip(model_b.get_weights(), model_c.get_weights()):
            np.testing.assert_allclose(w_loaded, w_full, rtol=0, atol=1e-5)

        # 4) Meta file exists
        meta_path = os.path.splitext(save_path)[0] + ".meta.json"
        assert os.path.exists(meta_path)
