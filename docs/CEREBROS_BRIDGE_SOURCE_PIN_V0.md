# Cerebros bridge source pin v0

This package boundary covers only the tracked `cerebros` core package. It
intentionally excludes `cerebrosllmutils`, `cerebros_dashboard`, and `notgpt`:
they have separate runtime surfaces and are not included in this reproducible
core artifact.

The legacy root and CI requirements files remain historical inputs only. In
particular, the legacy CI file contains a malformed fused requirement and is
not a dependency source for this package or its lock.

## Reproducibility target

- CPython 3.12 on Linux x86_64 with glibc 2.17 or newer.
- CPU-only import profile. `CUDA_VISIBLE_DEVICES=-1` and `JAX_PLATFORMS=cpu`
  are applied by the import-isolation test.
- The lock is the resolved dependency closure. Do not substitute an unpinned
  requirements file for it.

The core package imports TensorFlow, JAX, and NumPy directly in public modules,
so they are mandatory CPU-profile dependencies rather than optional extras.
There is deliberately no generic GPU extra: TensorFlow and JAX CUDA plugin
compatibility has not been demonstrated together.

## Optional integrations

- `orchestration`: Huey and Redis support for the excluded NotGPT orchestration
  surface. It is versioned here so the known undeclared Huey import has a
  declared, opt-in compatibility path; it does not activate a queue at import.
- `tracking`: MLflow integration, retained as an opt-in dependency because the
  core package does not import or start MLflow.
- `tuning`: Optuna integration, retained as an opt-in dependency because it is
  not imported by the core package.

## Verification

```bash
uv lock --python /usr/bin/python3.12
uv lock --locked
uv run --locked --group test pytest tests/test_import_side_effects.py
uv build
```

The import test runs each public `cerebros` module in a fresh child process and
fails if module import starts a thread or process, opens a socket, invokes a
subprocess, writes outside its isolated sandbox, or initializes a non-CPU
runtime through the test environment. Framework-local configuration files may
be created only inside that disposable sandbox; no global path is writable.

This package boundary is not a search campaign, oracle, holdout, or scientific
verdict authority.

## Current strict-proof status

The pinned CPU environment functionally imports all thirteen public core module
paths. The stricter child-process proof is intentionally still blocked: the
TensorFlow dependency path loads `h5py`, whose host-processor probe invokes a
short-lived `uname -p` subprocess during import. The isolation test fails on
that event rather than concealing it. This branch is therefore a reproducible
working base, not an accepted no-child-process import certificate.
