"""
Audit script: scans Keras models/layers in code to detect unregistered custom layers
that would break Keras 3 native serialization (.keras) round-trip.

Usage:
  python -m cerebros.persistence.audit_serialization

This performs a static inspection looking for subclasses of tf.keras.layers.Layer
that are not decorated with @tf.keras.utils.register_keras_serializable.

Note: heuristic-based; prioritize known layer modules.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import List

TARGET_DIRS = [
    os.path.join(os.path.dirname(__file__), ".."),
]
TARGET_DIRS = [os.path.abspath(p) for p in TARGET_DIRS]


@dataclass
class Finding:
    file: str
    line: int
    name: str
    message: str


def _is_layer_subclass(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Layer":
            return True
        if isinstance(base, ast.Name) and base.id == "Layer":
            return True
    return False


def _has_register_decorator(node: ast.ClassDef) -> bool:
    if not node.decorator_list:
        return False
    for dec in node.decorator_list:
        # @tf.keras.utils.register_keras_serializable(...)
        if isinstance(dec, ast.Attribute) and dec.attr == "register_keras_serializable":
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "register_keras_serializable":
            return True
        if isinstance(dec, ast.Name) and dec.id == "register_keras_serializable":
            return True
    return False


def scan_file(path: str) -> List[Finding]:
    findings: List[Finding] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
    except Exception:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_layer_subclass(node):
            if not _has_register_decorator(node):
                findings.append(
                    Finding(
                        file=path,
                        line=node.lineno,
                        name=node.name,
                        message="Layer subclass missing @register_keras_serializable",
                    )
                )
    return findings


def main() -> int:
    all_findings: List[Finding] = []
    for root in TARGET_DIRS:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                all_findings.extend(scan_file(path))

    if all_findings:
        print("Found potential serialization issues:")
        for f in all_findings:
            rel = os.path.relpath(f.file, os.getcwd())
            print(f"- {rel}:{f.line} {f.name}: {f.message}")
        return 1
    else:
        print("No unregistered custom layers detected.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
