"""
Build a PyTorch nn.Module graph from a neutral DAG spec exported by
cerebros.keras_export.export_keras_to_graph.

This is a minimal, extensible scaffold. It supports a subset of layers and
can be extended to more types as needed.
"""
from __future__ import annotations

from typing import Any, Dict, List
import json

try:  # Optional import so the repo still works without torch installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _dense_from_config(cfg: Dict[str, Any]) -> nn.Module:
    units = int(cfg.get("units", 1))
    use_bias = bool(cfg.get("use_bias", True))
    # in_features must be inferred at runtime; we create a LazyLinear
    layer = nn.LazyLinear(units, bias=use_bias)
    return layer


def _activation_from_config(class_name: str, cfg: Dict[str, Any]) -> nn.Module | None:
    act = cfg.get("activation") if isinstance(cfg, dict) else None
    if class_name == "Dense":
        # Keras Dense bakes activation inside; we split into Linear + Activation
        if act in (None, "linear", "identity"):
            return None
        return _map_activation(act)
    # Generic Activation layers
    if class_name in ("Activation",):
        if isinstance(cfg, dict):
            return _map_activation(cfg.get("activation", "linear"))
    return None


def _map_activation(name: str) -> nn.Module:
    name = (name or "").lower()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("elu",):
        return nn.ELU()
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("sigmoid",):
        return nn.Sigmoid()
    # default: identity
    return nn.Identity()


def _bn_from_config(cfg: Dict[str, Any]) -> nn.Module:
    momentum = cfg.get("momentum", 0.99)
    eps = cfg.get("epsilon", 1e-3)
    affine = True
    # Use LazyBatchNorm1d; infer num_features from input at first run
    return nn.LazyBatchNorm1d(eps=eps, momentum=momentum, affine=affine)


def _dropout_from_config(cfg: Dict[str, Any]) -> nn.Module:
    rate = float(cfg.get("rate", 0.5))
    return nn.Dropout(p=rate)


_LAYER_BUILDERS = {
    "Dense": _dense_from_config,
    "BatchNormalization": _bn_from_config,
    "Dropout": _dropout_from_config,
    # Add more: Conv1D, Conv2D, Add, Concatenate as dedicated ops in forward
}


class CerebrosTorchModule(nn.Module):
    def __init__(self, spec: Dict[str, Any]):
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch not available; cannot build Torch module.")

        self.spec = spec
        self.nodes = spec.get("nodes", [])
        # Register layers
        modules = {}
        for n in self.nodes:
            class_name = n.get("class_name")
            name = n.get("name")
            cfg = n.get("config", {})
            if class_name in ("Add", "Concatenate", "InputLayer"):
                # Handled in forward
                continue
            if class_name in _LAYERS_THAT_SPLIT_ACTIVATION():
                # For Dense, create Linear (+ optional activation)
                linear = _dense_from_config(cfg)
                self.add_module(name + "__linear", linear)
                act = _activation_from_config(class_name, cfg)
                if act is not None:
                    self.add_module(name + "__act", act)
                modules[name] = (class_name, cfg)
            elif class_name in _LAYER_BUILDERS:
                mod = _LAYER_BUILDERS[class_name](cfg)
                self.add_module(name, mod)
                modules[name] = (class_name, cfg)
            else:
                # Fallback to identity for unknown layers; extend as needed
                self.add_module(name, nn.Identity())
                modules[name] = (class_name, cfg)
        self._modules_meta = modules

    def forward(self, inputs: Dict[str, torch.Tensor] | torch.Tensor | List[torch.Tensor]):
        # Normalize inputs to a dict keyed by input layer names
        if isinstance(inputs, torch.Tensor):
            # Single input
            input_names = self.spec.get("inputs", ["input_0"]) or ["input_0"]
            tensors = {input_names[0]: inputs}
        elif isinstance(inputs, list):
            input_names = self.spec.get("inputs", [])
            tensors = {n: t for n, t in zip(input_names, inputs)}
        else:
            tensors = dict(inputs)

        for n in self.nodes:
            name = n["name"]
            class_name = n.get("class_name")
            cfg = n.get("config", {})
            ins = n.get("inputs", [])

            if class_name == "InputLayer":
                # Keras uses InputLayer; tensor is already provided
                continue

            # Resolve input tensors for this node
            xs: List[torch.Tensor] = []
            for up in ins:
                # Prefer the output tensor registered under the upstream node name
                if up in tensors:
                    xs.append(tensors[up])
                else:
                    # Some graphs may refer to tensor names; attempt to find by prefix
                    for k in list(tensors.keys()):
                        if k.startswith(up):
                            xs.append(tensors[k])
                            break
            x: torch.Tensor

            if class_name == "Add":
                assert len(xs) >= 2, "Add expects >=2 inputs"
                x = xs[0]
                for y in xs[1:]:
                    x = x + y
                tensors[name] = x
                continue

            if class_name == "Concatenate":
                axis = int(cfg.get("axis", 1)) if isinstance(cfg, dict) else 1
                x = torch.cat(xs, dim=axis)
                tensors[name] = x
                continue

            if class_name in _LAYERS_THAT_SPLIT_ACTIVATION():
                linear = getattr(self, name + "__linear")
                x = linear(xs[0] if len(xs) == 1 else torch.cat(xs, dim=1))
                act = getattr(self, name + "__act", None)
                if act is not None:
                    x = act(x)
                tensors[name] = x
                continue

            # Generic mapped layers
            mod = getattr(self, name, None)
            if mod is None:
                raise RuntimeError(f"Module for layer '{name}' not found")
            x = xs[0] if len(xs) == 1 else xs
            tensors[name] = mod(x)

        # Return outputs by declared names
        outs = self.spec.get("outputs", [])
        if not outs:
            # Fallback to last node
            last = self.nodes[-1]["name"]
            return tensors[last]
        if len(outs) == 1:
            return tensors[outs[0]]
        return [tensors[o] for o in outs]


def _LAYERS_THAT_SPLIT_ACTIVATION():
    # Keras Dense holds activation in its config; we emit Linear + Activation
    return {"Dense"}


def build_from_json(path: str) -> CerebrosTorchModule:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return CerebrosTorchModule(spec)

