import json
import os
import hashlib
import tensorflow as tf
from typing import Any, Dict, Optional


def save_model_and_meta(model: tf.keras.Model, path: str, meta: Dict[str, Any]) -> None:
    """Save a compiled Keras model to the .keras format alongside a meta JSON.

    The optimizer/compile state is preserved by default in Keras 3 when saving
    to the native .keras format. A sibling meta JSON is written next to it to
    capture run context (e.g., seeds, dataset hashes, etc.).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Ensure we save in the native Keras format by using a .keras extension
    if not path.endswith(".keras"):
        path = os.path.splitext(path)[0] + ".keras"
    # Save model with optimizer and loss/metrics state
    model.save(path)
    # Write meta JSON
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, indent=2)


def load_model_safe(path: str, custom_objects: Optional[dict] = None, safe_mode: bool = True) -> tf.keras.Model:
    """Load a Keras model from a .keras file using safe_mode by default.

    custom_objects is available for legacy models; ideally custom layers are
    registered via @keras.saving.register_keras_serializable.
    """
    return tf.keras.models.load_model(path, custom_objects=custom_objects, safe_mode=safe_mode)


def load_meta(path: str) -> Dict[str, Any]:
    """Load the sibling meta JSON for a given .keras path (or path stem)."""
    base = path
    if base.endswith(".keras"):
        base = os.path.splitext(base)[0]
    meta_path = base + ".meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def hash_tensor_batch(xs) -> str:
    sha = hashlib.sha256()
    def _upd(arr):
        import numpy as _np
        sha.update(_np.asarray(arr).tobytes())
    if isinstance(xs, (list, tuple)):
        for x in xs:
            _upd(x)
    else:
        _upd(xs)
    return sha.hexdigest()
