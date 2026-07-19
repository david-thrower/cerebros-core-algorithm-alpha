"""Clean-process import checks for the versioned Cerebros core package.

The public package deliberately excludes the legacy dashboard, LLM utilities,
and NotGPT orchestration surfaces.  Each core module is imported in a fresh
CPU-only child interpreter so an import cache cannot mask a side effect.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


CORE_PUBLIC_MODULES = (
    "cerebros",
    "cerebros.denseautomlstructuralcomponent",
    "cerebros.denseautomlstructuralcomponent.dense_automl_structural_component",
    "cerebros.levels",
    "cerebros.levels.levels",
    "cerebros.neuralnetworkfuture",
    "cerebros.neuralnetworkfuture.neural_network_future",
    "cerebros.nnfuturecomponent",
    "cerebros.nnfuturecomponent.neural_network_future_component",
    "cerebros.simplecerebrosrandomsearch",
    "cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search",
    "cerebros.units",
    "cerebros.units.units",
)

IMPORT_PROBE = textwrap.dedent(
    """
    import builtins
    import importlib
    import json
    import multiprocessing
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import threading

    module_name = sys.argv[1]
    sandbox = Path(os.environ["CEREBROS_IMPORT_SANDBOX"])
    events = []

    def forbidden(kind):
        def blocked(*args, **kwargs):
            events.append(kind)
            raise RuntimeError(f"forbidden import side effect: {kind}")
        return blocked

    original_open = builtins.open
    def writable_target_is_outside_sandbox(file):
        try:
            return not Path(file).resolve().is_relative_to(sandbox)
        except (OSError, TypeError):
            return True

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and writable_target_is_outside_sandbox(file):
            return forbidden("global_file_write")(file, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open
    original_socket = socket.socket
    class GuardedSocket(original_socket):
        def connect(self, *args, **kwargs):
            return forbidden("socket_connect")(*args, **kwargs)

        def connect_ex(self, *args, **kwargs):
            return forbidden("socket_connect")(*args, **kwargs)

        def bind(self, *args, **kwargs):
            return forbidden("socket_bind")(*args, **kwargs)

        def listen(self, *args, **kwargs):
            return forbidden("socket_listen")(*args, **kwargs)

    socket.socket = GuardedSocket
    socket.create_connection = forbidden("socket_connection")
    subprocess.Popen = forbidden("subprocess")
    os.system = forbidden("os_system")
    multiprocessing.Process.start = forbidden("process_start")
    threading.Thread.start = forbidden("thread_start")

    before = sorted(
        str(path.relative_to(sandbox))
        for path in sandbox.rglob("*")
        if path.is_file()
    )
    importlib.import_module(module_name)
    after = sorted(
        str(path.relative_to(sandbox))
        for path in sandbox.rglob("*")
        if path.is_file()
    )

    print(json.dumps({"events": events, "new_files": sorted(set(after) - set(before))}))
    """
)


@pytest.mark.parametrize("module_name", CORE_PUBLIC_MODULES)
def test_public_module_import_isolated(module_name: str, tmp_path: Path) -> None:
    sandbox = tmp_path / "isolated-import"
    sandbox.mkdir()
    environment = {
        "CEREBROS_IMPORT_SANDBOX": str(sandbox),
        "CUDA_VISIBLE_DEVICES": "-1",
        "HOME": str(sandbox / "home"),
        "JAX_PLATFORMS": "cpu",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(sandbox / "tmp"),
        "XDG_CACHE_HOME": str(sandbox / "cache"),
        "XDG_CONFIG_HOME": str(sandbox / "config"),
        "XDG_DATA_HOME": str(sandbox / "data"),
    }
    for directory in environment.values():
        if directory.startswith(str(sandbox)):
            Path(directory).mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, module_name],
        cwd=sandbox,
        env={**os.environ, **environment},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["events"] == []
    assert all(not Path(path).is_absolute() for path in result["new_files"])
