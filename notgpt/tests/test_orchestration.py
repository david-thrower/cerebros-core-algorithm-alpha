"""Test Huey task orchestration in immediate mode (no Redis)."""

import os

# Force immediate mode so tasks run inline without Redis
os.environ["HUEY_IMMEDIATE"] = "true"

from notgpt.orchestration.tasks import (
    preprocess_work_products,
    preprocess_qa,
    preprocess_references,
    train_stage_2,
    train_stage_3,
    train_stage_4,
)
from notgpt.orchestration.client import list_tasks


def _get(result_handle):
    """Extract result from Huey Result handle in immediate mode."""
    return result_handle(blocking=True, timeout=30)


def test_preprocess_work_products_task():
    result = _get(preprocess_work_products(1, num_samples=2))
    assert result["status"] == "complete"
    assert result["task"] == "work_products"


def test_preprocess_qa_task():
    result = _get(preprocess_qa(1, num_samples=2))
    assert result["status"] == "complete"
    assert result["task"] == "qa"


def test_train_stage_2():
    result = _get(train_stage_2(1))
    assert result["status"] == "complete"
    assert "checkpoint" in result


def test_train_stage_3():
    result = _get(train_stage_3(1, stage_2_checkpoint="s2_ckpt"))
    assert result["status"] == "complete"
    assert "checkpoint" in result


def test_train_stage_4():
    result = _get(train_stage_4(1, stage_3_checkpoint="s3_ckpt"))
    assert result["status"] == "complete"
    assert result["checkpoint"].startswith("stage_4_checkpoint")


def test_list_tasks():
    tasks = list_tasks()
    assert "preprocess_work_products" in tasks
    assert "train_stage_2" in tasks
    assert "run_full_pipeline" in tasks
    assert len(tasks) == 8
