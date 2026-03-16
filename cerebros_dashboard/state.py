"""Cerebros NotGPT dashboard state."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import reflex as rx

# Ensure notgpt pipeline modules are importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from notgpt.storage.db import get_engine, init_db, get_session
from notgpt.storage.models import (
    Colleague,
    ColleagueDocument,
    ColleagueQAPair,
    ColleagueSyntheticSample,
)
from notgpt.pipeline.text_extract import extract_text
from notgpt.pipeline.generators import Generators
from notgpt.pipeline import work_products, qa_upsampling, comm_threads, references

# Initialize once at module load
_engine = init_db()
_generators = Generators(use_llm=False, target_seq_len=500)


class AppState(rx.State):
    """Global app state — colleague list."""

    colleagues: list[dict] = []

    def load_colleagues(self):
        with get_session(_engine) as session:
            rows = (
                session.query(Colleague)
                .order_by(Colleague.created_at.desc())
                .all()
            )
            self.colleagues = [
                {"id": c.id, "name": c.name, "status": c.status}
                for c in rows
            ]


class WizardState(AppState):
    """Wizard state for 7-step colleague onboarding."""

    # Navigation
    step: int = 0

    # Step 1: Metadata
    colleague_id: int = 0
    colleague_name: str = ""
    colleague_desc: str = ""

    # Step 2: Work Products
    wp_files: list[dict] = []
    wp_processing: bool = False

    # Step 3: Q&A
    qa_pairs: list[dict] = []
    qa_prompt: str = ""
    qa_reasoning: str = ""
    qa_response: str = ""

    # Step 4: Comm Threads
    comm_platform: str = "Email"
    comm_identity: str = ""
    comm_files: list[dict] = []
    comm_processing: bool = False

    # Step 5: References
    ref_files: list[dict] = []
    ref_processing: bool = False

    # Step 6: Review
    synthetic_samples: list[dict] = []
    review_loading: bool = False

    # Step 7: Training
    training_status: str = "idle"
    training_stage: str = ""
    training_progress: int = 0

    # --- Navigation ---

    def next_step(self):
        if self.step < 7:
            self.step += 1

    def prev_step(self):
        if self.step > 0:
            self.step -= 1

    def go_to_step(self, step: int):
        self.step = step

    # --- Step 1: Create Colleague ---

    def save_colleague(self):
        if not self.colleague_name.strip():
            return rx.toast.error("Please enter a name")
        with get_session(_engine) as session:
            c = Colleague(
                name=self.colleague_name.strip(),
                description=self.colleague_desc.strip(),
            )
            session.add(c)
            session.flush()
            self.colleague_id = c.id
        self.next_step()
        return rx.toast.success(f"Created: {self.colleague_name}")

    # --- Step 2: Work Product Upload ---

    @rx.event
    async def handle_wp_upload(self, files: list[rx.UploadFile]):
        for file in files:
            data = await file.read()
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            text = extract_text(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            if not text.strip():
                yield rx.toast.warning(f"Could not extract text from {file.name}")
                continue

            with get_session(_engine) as session:
                doc = ColleagueDocument(
                    colleague_id=self.colleague_id,
                    category="work_product",
                    original_filename=file.name,
                    extracted_text=text,
                )
                session.add(doc)

            self.wp_files.append(
                {"name": file.name, "chars": len(text), "status": "uploaded"}
            )
            yield rx.toast.success(f"Uploaded: {file.name}")

    @rx.event(background=True)
    async def process_work_products(self):
        async with self:
            self.wp_processing = True
        await asyncio.to_thread(
            work_products.process, self.colleague_id, _engine, _generators
        )
        async with self:
            self.wp_processing = False
            self.step = 2

    def next_from_wp(self):
        self.step = 2
        return WizardState.process_work_products

    # --- Step 3: Q&A ---

    def add_qa_pair(self):
        if not self.qa_prompt.strip() or not self.qa_response.strip():
            return rx.toast.warning("Prompt and response required")
        with get_session(_engine) as session:
            qa = ColleagueQAPair(
                colleague_id=self.colleague_id,
                prompt=self.qa_prompt.strip(),
                reasoning=self.qa_reasoning.strip() or None,
                response=self.qa_response.strip(),
            )
            session.add(qa)
        self.qa_pairs.append(
            {
                "prompt": self.qa_prompt.strip()[:80],
                "response": self.qa_response.strip()[:80],
            }
        )
        self.qa_prompt = ""
        self.qa_reasoning = ""
        self.qa_response = ""
        return rx.toast.success("Q&A pair added")

    @rx.event(background=True)
    async def process_qa(self):
        async with self:
            pass  # just to get colleague_id
        await asyncio.to_thread(
            qa_upsampling.process, self.colleague_id, _engine, _generators
        )
        async with self:
            self.step = 3

    def next_from_qa(self):
        self.step = 3
        return WizardState.process_qa

    # --- Step 4: Comm Threads ---

    @rx.event
    async def handle_comm_upload(self, files: list[rx.UploadFile]):
        for file in files:
            data = await file.read()
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            text = extract_text(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            if not text.strip():
                yield rx.toast.warning(f"Could not extract text from {file.name}")
                continue

            with get_session(_engine) as session:
                doc = ColleagueDocument(
                    colleague_id=self.colleague_id,
                    category="communication",
                    original_filename=file.name,
                    extracted_text=text,
                    platform=self.comm_platform,
                    user_identity=self.comm_identity,
                )
                session.add(doc)

            self.comm_files.append({"name": file.name, "status": "uploaded"})
            yield rx.toast.success(f"Uploaded: {file.name}")

    @rx.event(background=True)
    async def process_comms(self):
        async with self:
            self.comm_processing = True
        await asyncio.to_thread(
            comm_threads.process, self.colleague_id, _engine, _generators
        )
        async with self:
            self.comm_processing = False
            self.step = 4

    def next_from_comms(self):
        self.step = 4
        return WizardState.process_comms

    # --- Step 5: References ---

    @rx.event
    async def handle_ref_upload(self, files: list[rx.UploadFile]):
        for file in files:
            data = await file.read()
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            text = extract_text(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            if not text.strip():
                yield rx.toast.warning(f"Could not extract text from {file.name}")
                continue

            with get_session(_engine) as session:
                doc = ColleagueDocument(
                    colleague_id=self.colleague_id,
                    category="reference",
                    original_filename=file.name,
                    extracted_text=text,
                )
                session.add(doc)

            self.ref_files.append({"name": file.name, "status": "uploaded"})
            yield rx.toast.success(f"Uploaded: {file.name}")

    @rx.event(background=True)
    async def process_refs(self):
        async with self:
            self.ref_processing = True
        await asyncio.to_thread(
            references.process, self.colleague_id, _engine, _generators
        )
        async with self:
            self.ref_processing = False
            self.step = 5

    def next_from_refs(self):
        self.step = 5
        return WizardState.process_refs

    # --- Step 6: Review ---

    def load_samples(self):
        self.review_loading = True
        with get_session(_engine) as session:
            samples = (
                session.query(ColleagueSyntheticSample)
                .filter_by(colleague_id=self.colleague_id)
                .order_by(ColleagueSyntheticSample.source_type)
                .all()
            )
            self.synthetic_samples = [
                {
                    "id": s.id,
                    "source": s.source_type,
                    "prompt": s.synthetic_prompt[:120],
                    "response": s.synthetic_response[:120],
                    "approved": s.approved,
                }
                for s in samples
            ]
        self.review_loading = False

    def approve_all(self):
        with get_session(_engine) as session:
            session.query(ColleagueSyntheticSample).filter_by(
                colleague_id=self.colleague_id
            ).update({"approved": True})
        self.load_samples()
        return rx.toast.success("All samples approved")

    def delete_sample(self, sample_id: int):
        with get_session(_engine) as session:
            session.query(ColleagueSyntheticSample).filter_by(id=sample_id).delete()
        self.load_samples()

    # --- Step 7: Training ---

    @rx.event(background=True)
    async def start_training(self):
        async with self:
            self.training_status = "running"
            self.training_progress = 0

        stages = [
            ("Stage 2: Domain Specialization", 33),
            ("Stage 3: General Instruct", 66),
            ("Stage 4: Personalization", 100),
        ]

        for stage_name, target_pct in stages:
            async with self:
                self.training_stage = stage_name

            # Simulate training (replace with real trainer call later)
            for p in range(self.training_progress, target_pct, 2):
                await asyncio.sleep(0.3)
                async with self:
                    self.training_progress = p

            async with self:
                self.training_progress = target_pct

        async with self:
            self.training_status = "complete"
            self.training_stage = "All stages complete"

        # Update colleague status
        with get_session(_engine) as session:
            c = session.get(Colleague, self.colleague_id)
            if c:
                c.status = "ready"

    # --- Reset ---

    def reset_wizard(self):
        self.step = 0
        self.colleague_id = 0
        self.colleague_name = ""
        self.colleague_desc = ""
        self.wp_files = []
        self.qa_pairs = []
        self.comm_files = []
        self.ref_files = []
        self.synthetic_samples = []
        self.training_status = "idle"
        self.training_progress = 0
        self.training_stage = ""
        self.load_colleagues()
