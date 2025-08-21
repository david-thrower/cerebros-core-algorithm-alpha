"""
Utilities to export a Keras Functional model into a neutral JSON graph spec
that can be used to rebuild the model in other frameworks (e.g., PyTorch).

Spec format (v0):
{
  "version": 0,
  "inputs": ["<input_tensor_name>", ...],
  "outputs": ["<output_tensor_name>", ...],
  "nodes": [
    {
      "name": "<layer_name>",
      "class_name": "Dense|BatchNormalization|Dropout|Add|Concatenate|InputLayer|...",
      "config": { ...layer.get_config()... },
      "inputs": ["<upstream_layer_name>", ...]
    },
    ...
  ]
}
"""
from __future__ import annotations

from typing import Any, Dict, List
import json

try:
    import tensorflow as tf  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    tf = None  # type: ignore


def _safe_layer_config(layer) -> Dict[str, Any]:
    try:
        return layer.get_config() or {}
    except Exception:
        return {}


def _tensor_name(t) -> str:
    try:
        # Tensor names look like 'dense/BiasAdd:0' — keep the op name
        return str(t.name).split(":")[0]
    except Exception:
        return str(t)


def export_keras_to_graph(model) -> Dict[str, Any]:
    """Export a tf.keras.Model to a minimal DAG spec.

    Requires a Functional or Model graph (not a pure Sequential with no names).
    """
    if tf is None:
        raise RuntimeError("TensorFlow not available; cannot export Keras model.")

    nodes: List[Dict[str, Any]] = []

    # Build list of nodes with inbound connections (layer order is already topological)
    for layer in model.layers:
        inputs: List[str] = []
        try:
            # Keras 2/TF 2.x
            inbound_nodes = getattr(layer, "_inbound_nodes", [])
            for node in inbound_nodes:
                in_layers = getattr(node, "inbound_layers", [])
                if not isinstance(in_layers, (list, tuple)):
                    in_layers = [in_layers]
                for in_l in in_layers:
                    if in_l is None:
                        continue
                    inputs.append(in_l.name)
        except Exception:
            pass

        # Deduplicate while preserving order
        seen = set()
        inputs = [x for x in inputs if not (x in seen or seen.add(x))]

        nodes.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "config": _safe_layer_config(layer),
                "inputs": inputs,
            }
        )

    inputs = []
    try:
        inputs = [str(getattr(t, "name", _tensor_name(t))).split(":")[0] for t in model.inputs]
    except Exception:
        pass

    outputs = []
    try:
        outputs = [str(getattr(t, "name", _tensor_name(t))).split(":")[0] for t in model.outputs]
    except Exception:
        pass

    spec: Dict[str, Any] = {
        "version": 0,
        "framework": "keras",
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
    }
    return spec


def save_graph_json(spec: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=False)


def export_and_save(model, path: str) -> None:
    spec = export_keras_to_graph(model)
    save_graph_json(spec, path)

