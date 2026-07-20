# Cerebros Import Isolation Remediation V0

## Decision

`CEREBROS_IMPORT_ISOLATION_REMEDIATION_V0` applies Strategy A to the pinned
source tree at `ef3650ecb22e68fb108e3dd0f9bc2f4073e43ccc`: TensorFlow and JAX
are no longer imported while Cerebros' 13 public modules are being inspected.
The backends load only when an explicitly invoked runtime operation first needs
them.

This is an import-time contract. It does not authorize or prove model
construction, training, inference, optimization, GPU execution, Campaign 002,
oracle or holdout access, hardware use, production activation, or scientific
verdicts.

## Runtime boundary

The import proof uses CPython 3.12.13 and the locked project wheel with:

- one-core CPU affinity;
- all declared native compute pools limited to one;
- CUDA and NVIDIA device visibility disabled;
- JAX constrained to the CPU platform;
- bytecode writes disabled;
- `HOME`, temporary paths, Keras, Matplotlib, Hugging Face, and XDG paths
  redirected beneath a private run root;
- socket connection, bind, listen, subprocess, process-start, shell, and
  thread-start operations denied by the import probe;
- an operating-system syscall trace over the aggregate locked-wheel import.

The host does not permit unprivileged network namespaces. The proof therefore
does not claim namespace isolation. It proves the narrower strict property
required by this lane: the exact import performed no network syscall, created
no child process or thread, and opened no path for writing. The Python guard
also fails the run if the package attempts the corresponding high-level
operations.

## Compatibility rule

Deferred defaults use an internal sentinel. Omitting a TensorFlow-backed
default still resolves the historical default when the owning object is
constructed, while an explicit `None` remains `None`. This avoids importing a
backend merely to evaluate a function signature and avoids silently changing
the meaning of explicit caller input.

## Verification

Focused source-tree tests:

```text
15 passed in 9.82s
```

Locked-wheel clean-room matrix:

```text
isolated module imports: 26/26
aggregate imports:       2/2
Python:                  CPython 3.12.13
CPU affinity:            1
threads before/after:    1/1
backend modules loaded:  0
child processes:         0
sandbox writes:          0
```

Final aggregate syscall trace:

```text
socket/connect/bind/listen calls: 0
clone/clone3/fork/vfork calls:     0
writable open/create/truncate:     0
```

Immutable host-local proof anchors:

```text
wheel sha256:
  45fb3fb9a4647de16fe6237b616104125d1621040df0d2e4c2f2c96ea24cfaa4
uv.lock sha256:
  1c646185bd8830d8d4231f6d923f83be28b7a6941ded2afd3706d20d05482ebd
exported lock sha256:
  ac3773ef5681e955cdb7585cb0b1863195a3c0f0936671093023b33e30866bde
syscall trace sha256:
  0c4be3ee7930dfb29c340daee816c6b9961b708ee743aadac9f23558a782528b
clean-room probe sha256:
  33a08f097e5c47240e2126125527dd9189915d2cdfc1080802f512bfefaef3eb
clean-room matrix runner sha256:
  cc91608a527db5f4c870244deca9642f980b28bf27f8dd8c2d44f00bc8000d1d
```

The large trace and clean-room runner remain host-local. The committed receipt
contains their hashes but no host path, secret, capability payload, or runtime
authority.

## Claim

For this exact source tree and locked wheel:

```text
deferred_backend_imports=true
public_module_imports=13
persistent_child_processes=0
persistent_listeners=0
network_side_effects=0
writes_during_import=0
native_thread_count_after_import=1
source_runtime_changes=5
dependency_changes=0
production_activation=false
```

The prior strict-import BLOCK remains valid for the parent source-pin commit.
This candidate is a new remediation result; it does not rewrite that history.
