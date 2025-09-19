"""Quick demo comparing Dense vs VoxelQuantizedOutput.

Run from repo root after activating venv:
    python -m documentation.examples.voxel_output_demo
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import keras
except Exception:  # fallback to TF Keras
    from tensorflow import keras  # type: ignore

import numpy as np
import tensorflow as tf

from cerebros.layers import VoxelQuantizedOutput


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 128)).astype("float32")
    y = (rng.random(size=(32, 10)) > 0.5).astype("float32")

    dense = keras.Sequential([
        keras.layers.Input(shape=(128,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid"),
    ])

    voxel = keras.Sequential([
        keras.layers.Input(shape=(128,)),
        keras.layers.Dense(64, activation="relu"),
        VoxelQuantizedOutput(10, n_bits=4, activation="sigmoid", signed=False),
    ])

    dense.compile(optimizer="adam", loss="binary_crossentropy")
    voxel.compile(optimizer="adam", loss="binary_crossentropy")

    # single forward/backward to validate graph
    dense.train_on_batch(x, y)
    voxel.train_on_batch(x, y)

    d_out = dense.predict(x, verbose=0)
    v_out = voxel.predict(x, verbose=0)

    print("Dense output shape:", d_out.shape)
    print("Voxel output shape:", v_out.shape)

    # Parameter counts (rough)
    dense_params = dense.count_params()
    voxel_params = voxel.count_params()
    print("Dense params:", dense_params)
    print("Voxel params:", voxel_params)


if __name__ == "__main__":
    main()
