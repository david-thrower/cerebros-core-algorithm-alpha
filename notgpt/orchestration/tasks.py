"""Huey task definitions for all Cerebros pipelines.

Each task is a self-contained unit of work that runs in a worker pod.
KEDA monitors the Redis queue length and scales worker pods 0→N.

Tasks:
    Preprocessing (run on CPU pods):
        - preprocess_work_products
        - preprocess_qa
        - preprocess_comm_threads
        - preprocess_references

    Training (run on GPU pods):
        - train_stage_2  (domain specialization)
        - train_stage_3  (general instruct)
        - train_stage_4  (user personalization)

    Orchestration:
        - run_full_pipeline  (chains preprocessing → training)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from notgpt.orchestration.queue import huey

# Ensure imports work
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ─────────────────────────────────────────────
# Preprocessing Tasks (CPU)
# ─────────────────────────────────────────────

@huey.task(retries=2)
def preprocess_work_products(colleague_id: int, num_samples: int = 3) -> dict:
    """Process work product documents into synthetic instruct samples."""
    from notgpt.storage.db import get_engine, init_db
    from notgpt.pipeline.generators import Generators
    from notgpt.pipeline import work_products

    engine = init_db()
    generators = Generators(use_llm=False, target_seq_len=500)
    work_products.process(colleague_id, engine, generators, num_samples)
    return {"status": "complete", "task": "work_products", "colleague_id": colleague_id}


@huey.task(retries=2)
def preprocess_qa(colleague_id: int, num_samples: int = 3) -> dict:
    """Upsample Q&A pairs into synthetic variants."""
    from notgpt.storage.db import get_engine, init_db
    from notgpt.pipeline.generators import Generators
    from notgpt.pipeline import qa_upsampling

    engine = init_db()
    generators = Generators(use_llm=False, target_seq_len=500)
    qa_upsampling.process(colleague_id, engine, generators, num_samples)
    return {"status": "complete", "task": "qa", "colleague_id": colleague_id}


@huey.task(retries=2)
def preprocess_comm_threads(colleague_id: int, num_samples: int = 3) -> dict:
    """Process communication threads into instruct samples."""
    from notgpt.storage.db import get_engine, init_db
    from notgpt.pipeline.generators import Generators
    from notgpt.pipeline import comm_threads

    engine = init_db()
    generators = Generators(use_llm=False, target_seq_len=500)
    comm_threads.process(colleague_id, engine, generators, num_samples)
    return {"status": "complete", "task": "comm_threads", "colleague_id": colleague_id}


@huey.task(retries=2)
def preprocess_references(colleague_id: int, num_samples: int = 3) -> dict:
    """Process reference documents into pretraining samples."""
    from notgpt.storage.db import get_engine, init_db
    from notgpt.pipeline.generators import Generators
    from notgpt.pipeline import references

    engine = init_db()
    generators = Generators(use_llm=False, target_seq_len=500)
    references.process(colleague_id, engine, generators, num_samples)
    return {"status": "complete", "task": "references", "colleague_id": colleague_id}


# ─────────────────────────────────────────────
# Training Tasks (GPU)
# ─────────────────────────────────────────────

@huey.task(retries=1)
def train_stage_2(
    colleague_id: int,
    base_checkpoint: str = "stage_1_partially_pretrained_model",
) -> dict:
    """Stage 2: Domain specialization.

    Inputs:
        - stage_2_domain_specialization_relevant_subset
        - stage_2_domain_specialization_general_subset
        - user_provided_internal_references_synthetic_dataset
        - stage_1_partially_pretrained_model_checkpoint

    Output: stage_2_model_checkpoint

    TODO: Wire to real trainer (train_a_generative_llm.py) when ready.
    """
    # Placeholder — replace with actual training call
    # from cerebros_training import train_model
    # checkpoint = train_model(
    #     base_checkpoint=base_checkpoint,
    #     dataset_ids=["stage_2_relevant", "stage_2_general", "user_refs"],
    #     colleague_id=colleague_id,
    #     stage="stage_2",
    # )
    time.sleep(2)  # Simulate
    return {
        "status": "complete",
        "task": "train_stage_2",
        "colleague_id": colleague_id,
        "checkpoint": f"stage_2_checkpoint_{colleague_id}",
    }


@huey.task(retries=1)
def train_stage_3(
    colleague_id: int,
    stage_2_checkpoint: str = "",
) -> dict:
    """Stage 3: General instruct fine-tuning.

    Inputs:
        - stage_3_general_instruct_relevant_subset
        - stage_3_general_instruct_general_subset
        - stage_2_model_checkpoint

    Output: stage_3_model_checkpoint
    """
    time.sleep(2)  # Simulate
    return {
        "status": "complete",
        "task": "train_stage_3",
        "colleague_id": colleague_id,
        "checkpoint": f"stage_3_checkpoint_{colleague_id}",
    }


@huey.task(retries=1)
def train_stage_4(
    colleague_id: int,
    stage_3_checkpoint: str = "",
) -> dict:
    """Stage 4: User personalization fine-tuning.

    Inputs:
        - stage_4_user_provided_instruct_relevant_subset
        - stage_4_user_provided_instruct_general_subset
        - stage_3_model_checkpoint

    Output: stage_4_model_checkpoint (final personalized model)
    """
    time.sleep(2)  # Simulate
    return {
        "status": "complete",
        "task": "train_stage_4",
        "colleague_id": colleague_id,
        "checkpoint": f"stage_4_checkpoint_{colleague_id}",
    }


# ─────────────────────────────────────────────
# Orchestration — Full Pipeline
# ─────────────────────────────────────────────

@huey.task(retries=0)
def run_full_pipeline(colleague_id: int, num_samples: int = 3) -> dict:
    """Run the complete pipeline: preprocess → train stages 2→3→4.

    This is the top-level task dispatched when the user clicks "Start Training".
    It chains preprocessing and training in sequence.
    """
    from notgpt.storage.db import get_engine, init_db, get_session
    from notgpt.storage.models import Colleague

    engine = init_db()

    def update_status(status: str):
        with get_session(engine) as session:
            c = session.get(Colleague, colleague_id)
            if c:
                c.status = status

    # --- Phase 1: Preprocessing ---
    update_status("preprocessing")

    wp_result = preprocess_work_products(colleague_id, num_samples)
    wp_result(blocking=True, timeout=600)

    qa_result = preprocess_qa(colleague_id, num_samples)
    qa_result(blocking=True, timeout=600)

    comm_result = preprocess_comm_threads(colleague_id, num_samples)
    comm_result(blocking=True, timeout=600)

    ref_result = preprocess_references(colleague_id, num_samples)
    ref_result(blocking=True, timeout=600)

    # --- Phase 2: Training ---
    update_status("training_stage_2")
    s2 = train_stage_2(colleague_id)
    s2_result = s2(blocking=True, timeout=7200)

    update_status("training_stage_3")
    s3 = train_stage_3(colleague_id, s2_result["checkpoint"])
    s3_result = s3(blocking=True, timeout=7200)

    update_status("training_stage_4")
    s4 = train_stage_4(colleague_id, s3_result["checkpoint"])
    s4_result = s4(blocking=True, timeout=7200)

    # --- Done ---
    update_status("ready")

    return {
        "status": "complete",
        "colleague_id": colleague_id,
        "final_checkpoint": s4_result["checkpoint"],
    }
