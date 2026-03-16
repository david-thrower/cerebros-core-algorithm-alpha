"""RPC client for dispatching tasks from the dashboard.

Usage from the Reflex app:
    from notgpt.orchestration.client import dispatch, get_result

    # Fire and forget
    task_id = dispatch("preprocess_work_products", colleague_id=123)

    # Check status later
    result = get_result(task_id, timeout=5)
    if result is not None:
        print(result)  # {"status": "complete", ...}

    # Or dispatch the full pipeline
    task_id = dispatch("run_full_pipeline", colleague_id=123)
"""

from __future__ import annotations

from typing import Any

from notgpt.orchestration.tasks import (
    preprocess_work_products,
    preprocess_qa,
    preprocess_comm_threads,
    preprocess_references,
    train_stage_2,
    train_stage_3,
    train_stage_4,
    run_full_pipeline,
)

_TASK_MAP = {
    "preprocess_work_products": preprocess_work_products,
    "preprocess_qa": preprocess_qa,
    "preprocess_comm_threads": preprocess_comm_threads,
    "preprocess_references": preprocess_references,
    "train_stage_2": train_stage_2,
    "train_stage_3": train_stage_3,
    "train_stage_4": train_stage_4,
    "run_full_pipeline": run_full_pipeline,
}


def dispatch(task_name: str, **kwargs: Any) -> str:
    """Dispatch a task to the Huey queue. Returns the task ID."""
    task_fn = _TASK_MAP.get(task_name)
    if task_fn is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(_TASK_MAP)}")
    result = task_fn(**kwargs)
    return result.id


def get_result(task_id: str, timeout: float = 0) -> dict | None:
    """Poll for a task result. Returns None if not ready yet.

    Args:
        task_id: The ID returned by dispatch().
        timeout: How long to wait (0 = don't block, just check).

    Returns:
        The task result dict, or None if still pending.
    """
    from huey.api import Result

    # Reconstruct the result handle from the task ID
    result = Result(huey=preprocess_work_products.huey, task=None)
    result._task_id = task_id

    try:
        if timeout > 0:
            return result.get(blocking=True, timeout=timeout)
        else:
            return result.get(blocking=False)
    except Exception:
        return None


def list_tasks() -> list[str]:
    """List available task names."""
    return list(_TASK_MAP.keys())
