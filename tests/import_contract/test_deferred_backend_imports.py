"""Strategy-A contract tests for backend-free Cerebros module imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PUBLIC_MODULES = (
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

BACKEND_PREFIXES = ("jax", "jaxlib", "tensorflow")

AGGREGATE_PROBE = textwrap.dedent(
    """
    import importlib
    import json
    import multiprocessing
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import threading

    module_names = sys.argv[1:]
    backend_prefixes = ("jax", "jaxlib", "tensorflow")
    sandbox = Path(os.environ["CEREBROS_IMPORT_SANDBOX"])
    events = []

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    def forbidden(kind):
        def blocked(*args, **kwargs):
            events.append(kind)
            raise RuntimeError(f"forbidden import side effect: {kind}")
        return blocked

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
    original_popen = subprocess.Popen
    class GuardedPopen(original_popen):
        def __init__(self, *args, **kwargs):
            forbidden("subprocess")(*args, **kwargs)

    subprocess.Popen = GuardedPopen
    os.system = forbidden("os_system")
    multiprocessing.Process.start = forbidden("process_start")
    threading.Thread.start = forbidden("thread_start")

    before_files = sorted(
        str(path.relative_to(sandbox))
        for path in sandbox.rglob("*")
        if path.is_file()
    )
    thread_count_before = len(list(Path("/proc/self/task").iterdir()))

    for module_name in module_names:
        importlib.import_module(module_name)

    thread_count_after = len(list(Path("/proc/self/task").iterdir()))
    child_file = Path(f"/proc/self/task/{os.getpid()}/children")
    child_processes_remaining = child_file.read_text(encoding="utf-8").split()
    backend_modules = sorted(
        name for name in sys.modules
        if name in backend_prefixes
        or name.startswith(tuple(f"{prefix}." for prefix in backend_prefixes))
    )
    deferred_module = importlib.import_module(
        "cerebros.denseautomlstructuralcomponent.dense_automl_structural_component"
    )
    proxy_loaded = deferred_module.jnp._module is not None
    after_files = sorted(
        str(path.relative_to(sandbox))
        for path in sandbox.rglob("*")
        if path.is_file()
    )

    print(json.dumps({
        "backend_modules": backend_modules,
        "child_processes_remaining": child_processes_remaining,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "events": events,
        "new_files": sorted(set(after_files) - set(before_files)),
        "proxy_loaded": proxy_loaded,
        "thread_count_after": thread_count_after,
        "thread_count_before": thread_count_before,
    }, sort_keys=True))
    """
)


def _bounded_environment(sandbox: Path) -> dict[str, str]:
    environment = {
        "BLIS_NUM_THREADS": "1",
        "CEREBROS_IMPORT_SANDBOX": str(sandbox),
        "CUDA_VISIBLE_DEVICES": "-1",
        "HOME": str(sandbox / "home"),
        "HF_HOME": str(sandbox / "huggingface"),
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_PLATFORMS": "cpu",
        "KERAS_HOME": str(sandbox / "keras"),
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(sandbox / "matplotlib"),
        "NVIDIA_VISIBLE_DEVICES": "none",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "RAYON_NUM_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TMP": str(sandbox / "tmp"),
        "TMPDIR": str(sandbox / "tmp"),
        "TEMP": str(sandbox / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "VECLIB_MAXIMUM_THREADS": "1",
        "XDG_CACHE_HOME": str(sandbox / "cache"),
        "XDG_CONFIG_HOME": str(sandbox / "config"),
        "XDG_DATA_HOME": str(sandbox / "data"),
    }
    for value in environment.values():
        if value.startswith(str(sandbox)):
            Path(value).mkdir(parents=True, exist_ok=True)
    return {**os.environ, **environment}


@pytest.mark.parametrize("repetition", range(2))
def test_aggregate_import_defers_backends(
    repetition: int, tmp_path: Path
) -> None:
    sandbox = tmp_path / f"aggregate-{repetition}"
    sandbox.mkdir()

    completed = subprocess.run(
        [sys.executable, "-c", AGGREGATE_PROBE, *PUBLIC_MODULES],
        cwd=sandbox,
        env=_bounded_environment(sandbox),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "backend_modules": [],
        "child_processes_remaining": [],
        "cpu_affinity_count": 1,
        "events": [],
        "new_files": [],
        "proxy_loaded": False,
        "thread_count_after": 1,
        "thread_count_before": 1,
    }
