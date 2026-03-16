# notgpt/app.py
from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path

from nicegui import ui, app, background_tasks

from notgpt.storage.db import get_engine, init_db, get_session
from notgpt.storage.models import (
    Colleague, ColleagueDocument, ColleagueQAPair, ColleagueSyntheticSample,
)
from notgpt.pipeline.text_extract import extract_text
from notgpt.pipeline.generators import Generators
from notgpt.pipeline import work_products, qa_upsampling, comm_threads, references

engine = None
generators = None


def startup():
    global engine, generators
    engine = init_db()
    generators = Generators(use_llm=False, target_seq_len=500)


app.on_startup(startup)


# ============================================================
# Page: Splash (list colleagues, create new)
# ============================================================

@ui.page("/")
def splash_page():
    ui.label("NotGPT — AI Colleague Studio").classes(
        "text-3xl font-bold mb-6"
    )

    with get_session(engine) as session:
        colleagues = (
            session.query(Colleague)
            .order_by(Colleague.created_at.desc())
            .all()
        )
        rows = [
            {"id": c.id, "name": c.name, "status": c.status}
            for c in colleagues
        ]

    if rows:
        ui.table(
            columns=[
                {"name": "name", "label": "Name", "field": "name", "align": "left"},
                {"name": "status", "label": "Status", "field": "status"},
            ],
            rows=rows,
        ).classes("w-full max-w-2xl")
    else:
        ui.label("No colleagues yet. Create your first one!").classes(
            "text-gray-500 italic"
        )

    ui.button(
        "Create New Colleague",
        on_click=lambda: ui.navigate.to("/new"),
        icon="add",
    ).classes("mt-6")


# ============================================================
# Page: 7-Step Wizard
# ============================================================

@ui.page("/new")
def wizard_page():
    state = {"colleague_id": None, "processing": False}

    ui.label("Create a New AI Colleague").classes("text-2xl font-bold mb-4")

    with ui.stepper().props("vertical").classes("w-full max-w-4xl") as stepper:

        # ------ Step 1: Metadata ------
        with ui.step("Name Your Colleague"):
            ui.label(
                "Give your AI colleague a name and optional description."
            ).classes("text-gray-600 mb-2")
            name_input = ui.input(
                "Name", placeholder="e.g. Sales Assistant"
            ).classes("w-full max-w-md")
            desc_input = ui.textarea(
                "Description (optional)",
                placeholder="What will this colleague help with?",
            ).classes("w-full max-w-md")

            with ui.stepper_navigation():
                def save_metadata():
                    name = name_input.value.strip()
                    if not name:
                        ui.notify("Please enter a name", type="warning")
                        return
                    with get_session(engine) as session:
                        c = Colleague(
                            name=name,
                            description=desc_input.value.strip(),
                        )
                        session.add(c)
                        session.flush()
                        state["colleague_id"] = c.id
                    ui.notify(f"Created: {name}", type="positive")
                    stepper.next()

                ui.button("Next", on_click=save_metadata)

        # ------ Step 2: Work Products ------
        with ui.step("Upload Work Products"):
            ui.label(
                "Upload examples of your completed deliverables — reports, "
                "invoices, briefs, documents. Upload as many as possible."
            ).classes("text-gray-600 mb-2")

            wp_status = ui.label("").classes("text-sm text-gray-500")

            async def handle_wp_upload(e):
                if not state["colleague_id"]:
                    ui.notify("Create colleague first", type="warning")
                    return
                # Save uploaded file to temp, extract text
                content = e.content.read()
                with tempfile.NamedTemporaryFile(
                    suffix=Path(e.name).suffix, delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                text = extract_text(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)

                if not text.strip():
                    ui.notify(f"Could not extract text from {e.name}", type="warning")
                    return

                with get_session(engine) as session:
                    doc = ColleagueDocument(
                        colleague_id=state["colleague_id"],
                        category="work_product",
                        original_filename=e.name,
                        extracted_text=text,
                    )
                    session.add(doc)

                wp_status.text = f"Uploaded: {e.name} ({len(text)} chars)"
                ui.notify(f"Uploaded {e.name}", type="positive")

            ui.upload(
                on_upload=handle_wp_upload,
                multiple=True,
                label="Drop files here or click to upload",
            ).props('accept=".pdf,.docx,.doc,.txt,.md"').classes("w-full max-w-md")

            with ui.stepper_navigation():
                async def process_wp_and_next():
                    if state["colleague_id"]:
                        ui.notify("Processing work products...", type="info")
                        await asyncio.to_thread(
                            work_products.process,
                            state["colleague_id"],
                            engine,
                            generators,
                        )
                        ui.notify("Work products processed!", type="positive")
                    stepper.next()

                ui.button("Next", on_click=process_wp_and_next)
                ui.button("Back", on_click=stepper.previous).props("flat")

        # ------ Step 3: Q&A Pairs ------
        with ui.step("Questions & Answers"):
            ui.label(
                "Add examples of questions you get asked and how you answer them. "
                "Think of it as explaining your job to an intern."
            ).classes("text-gray-600 mb-2")

            qa_rows = []
            qa_table = ui.table(
                columns=[
                    {"name": "prompt", "label": "Question/Prompt", "field": "prompt", "align": "left"},
                    {"name": "response", "label": "Answer/Response", "field": "response", "align": "left"},
                ],
                rows=qa_rows,
            ).classes("w-full max-w-3xl")

            with ui.row().classes("gap-2 mt-2"):
                qa_prompt = ui.input("Prompt/Question").classes("flex-grow")
                qa_reasoning = ui.input("Reasoning (optional)").classes("flex-grow")
                qa_response = ui.input("Response/Answer").classes("flex-grow")

                def add_qa():
                    if not qa_prompt.value.strip() or not qa_response.value.strip():
                        ui.notify("Prompt and response are required", type="warning")
                        return
                    if not state["colleague_id"]:
                        ui.notify("Create colleague first", type="warning")
                        return

                    with get_session(engine) as session:
                        pair = ColleagueQAPair(
                            colleague_id=state["colleague_id"],
                            prompt=qa_prompt.value.strip(),
                            reasoning=qa_reasoning.value.strip() or None,
                            response=qa_response.value.strip(),
                        )
                        session.add(pair)

                    qa_rows.append({
                        "prompt": qa_prompt.value.strip()[:80],
                        "response": qa_response.value.strip()[:80],
                    })
                    qa_table.update()
                    qa_prompt.value = ""
                    qa_reasoning.value = ""
                    qa_response.value = ""
                    ui.notify("Q&A pair added", type="positive")

                ui.button("Add", on_click=add_qa, icon="add")

            with ui.stepper_navigation():
                async def process_qa_and_next():
                    if state["colleague_id"]:
                        ui.notify("Processing Q&A pairs...", type="info")
                        await asyncio.to_thread(
                            qa_upsampling.process,
                            state["colleague_id"],
                            engine,
                            generators,
                        )
                        ui.notify("Q&A processed!", type="positive")
                    stepper.next()

                ui.button("Next", on_click=process_qa_and_next)
                ui.button("Back", on_click=stepper.previous).props("flat")

        # ------ Step 4: Communication Threads ------
        with ui.step("Communication Threads"):
            ui.label(
                "Upload examples of your email threads, Slack messages, "
                "or Discord conversations."
            ).classes("text-gray-600 mb-2")

            platform_select = ui.select(
                ["Email", "Slack", "Discord", "Other"],
                value="Email",
                label="Platform",
            ).classes("w-48")
            identity_input = ui.input(
                "Your identity (email/handle)",
                placeholder="you@company.com",
            ).classes("w-full max-w-md")

            comm_status = ui.label("").classes("text-sm text-gray-500")

            async def handle_comm_upload(e):
                if not state["colleague_id"]:
                    ui.notify("Create colleague first", type="warning")
                    return
                content = e.content.read()
                with tempfile.NamedTemporaryFile(
                    suffix=Path(e.name).suffix, delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                text = extract_text(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)

                if not text.strip():
                    ui.notify(f"Could not extract text from {e.name}", type="warning")
                    return

                with get_session(engine) as session:
                    doc = ColleagueDocument(
                        colleague_id=state["colleague_id"],
                        category="communication",
                        original_filename=e.name,
                        extracted_text=text,
                        platform=platform_select.value,
                        user_identity=identity_input.value.strip(),
                    )
                    session.add(doc)

                comm_status.text = f"Uploaded: {e.name}"
                ui.notify(f"Uploaded {e.name}", type="positive")

            ui.upload(
                on_upload=handle_comm_upload,
                multiple=True,
                label="Drop thread exports here",
            ).props('accept=".txt,.md,.csv,.eml"').classes("w-full max-w-md")

            with ui.stepper_navigation():
                async def process_comm_and_next():
                    if state["colleague_id"]:
                        ui.notify("Processing threads...", type="info")
                        await asyncio.to_thread(
                            comm_threads.process,
                            state["colleague_id"],
                            engine,
                            generators,
                        )
                        ui.notify("Threads processed!", type="positive")
                    stepper.next()

                ui.button("Next", on_click=process_comm_and_next)
                ui.button("Back", on_click=stepper.previous).props("flat")

        # ------ Step 5: Internal References ------
        with ui.step("Internal References"):
            ui.label(
                "Upload manuals, SOPs, policies, law books — any reference "
                "materials you use to do your job."
            ).classes("text-gray-600 mb-2")

            ref_status = ui.label("").classes("text-sm text-gray-500")

            async def handle_ref_upload(e):
                if not state["colleague_id"]:
                    ui.notify("Create colleague first", type="warning")
                    return
                content = e.content.read()
                with tempfile.NamedTemporaryFile(
                    suffix=Path(e.name).suffix, delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                text = extract_text(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)

                if not text.strip():
                    ui.notify(f"Could not extract text from {e.name}", type="warning")
                    return

                with get_session(engine) as session:
                    doc = ColleagueDocument(
                        colleague_id=state["colleague_id"],
                        category="reference",
                        original_filename=e.name,
                        extracted_text=text,
                    )
                    session.add(doc)

                ref_status.text = f"Uploaded: {e.name}"
                ui.notify(f"Uploaded {e.name}", type="positive")

            ui.upload(
                on_upload=handle_ref_upload,
                multiple=True,
                label="Drop reference documents here",
            ).props('accept=".pdf,.docx,.doc,.txt,.md"').classes("w-full max-w-md")

            with ui.stepper_navigation():
                async def process_refs_and_next():
                    if state["colleague_id"]:
                        ui.notify("Processing references...", type="info")
                        await asyncio.to_thread(
                            references.process,
                            state["colleague_id"],
                            engine,
                            generators,
                        )
                        ui.notify("References processed!", type="positive")
                    stepper.next()

                ui.button("Next", on_click=process_refs_and_next)
                ui.button("Back", on_click=stepper.previous).props("flat")

        # ------ Step 6: Review Synthetic Data ------
        with ui.step("Review Synthetic Data"):
            ui.label(
                "Review the synthetic training samples generated from your data. "
                "Edit or delete any that don't look right, then approve."
            ).classes("text-gray-600 mb-2")

            review_container = ui.column().classes("w-full")

            def load_review_data():
                review_container.clear()
                with review_container:
                    if not state["colleague_id"]:
                        ui.label("No colleague created yet").classes("text-gray-500")
                        return

                    with get_session(engine) as session:
                        samples = (
                            session.query(ColleagueSyntheticSample)
                            .filter_by(colleague_id=state["colleague_id"])
                            .order_by(ColleagueSyntheticSample.source_type)
                            .all()
                        )
                        sample_data = [
                            {
                                "id": s.id,
                                "source": s.source_type,
                                "prompt": s.synthetic_prompt[:100],
                                "response": s.synthetic_response[:100],
                                "approved": "Yes" if s.approved else "No",
                            }
                            for s in samples
                        ]

                    if not sample_data:
                        ui.label(
                            "No synthetic samples yet. Go back and upload some data first."
                        ).classes("text-gray-500")
                        return

                    ui.label(f"{len(sample_data)} synthetic samples generated").classes(
                        "text-lg font-semibold mb-2"
                    )

                    ui.table(
                        columns=[
                            {"name": "source", "label": "Source", "field": "source"},
                            {"name": "prompt", "label": "Synthetic Prompt", "field": "prompt", "align": "left"},
                            {"name": "response", "label": "Response (truncated)", "field": "response", "align": "left"},
                            {"name": "approved", "label": "Approved", "field": "approved"},
                        ],
                        rows=sample_data,
                        pagination={"rowsPerPage": 10},
                    ).classes("w-full")

                    def approve_all():
                        with get_session(engine) as session:
                            session.query(ColleagueSyntheticSample).filter_by(
                                colleague_id=state["colleague_id"]
                            ).update({"approved": True})
                        ui.notify(
                            f"Approved all {len(sample_data)} samples",
                            type="positive",
                        )
                        load_review_data()

                    ui.button(
                        "Approve All Samples",
                        on_click=approve_all,
                        icon="check_circle",
                    ).classes("mt-4").props("color=positive")

            # Load data when step becomes active
            load_review_data()

            with ui.stepper_navigation():
                ui.button("Next", on_click=stepper.next)
                ui.button("Back", on_click=stepper.previous).props("flat")

        # ------ Step 7: Training ------
        with ui.step("Start Training"):
            ui.label(
                "Review the summary below and start training your AI Colleague."
            ).classes("text-gray-600 mb-2")

            summary_container = ui.column().classes("w-full")
            progress_label = ui.label("").classes("text-lg font-semibold mt-4")

            def load_summary():
                summary_container.clear()
                with summary_container:
                    if not state["colleague_id"]:
                        ui.label("No colleague").classes("text-gray-500")
                        return

                    with get_session(engine) as session:
                        approved = (
                            session.query(ColleagueSyntheticSample)
                            .filter_by(
                                colleague_id=state["colleague_id"],
                                approved=True,
                            )
                            .count()
                        )
                        total = (
                            session.query(ColleagueSyntheticSample)
                            .filter_by(colleague_id=state["colleague_id"])
                            .count()
                        )
                        colleague = session.get(Colleague, state["colleague_id"])
                        name = colleague.name if colleague else "Unknown"
                        status = colleague.status if colleague else "unknown"

                    ui.label(f"Colleague: {name}").classes("text-xl font-bold")
                    ui.label(f"Status: {status}").classes("text-gray-600")
                    ui.label(f"Approved samples: {approved} / {total}")

                    if approved == 0:
                        ui.label(
                            "No approved samples. Go back to Step 6 and approve."
                        ).classes("text-orange-600 mt-2")

            load_summary()

            async def start_training():
                if not state["colleague_id"]:
                    return
                with get_session(engine) as session:
                    c = session.get(Colleague, state["colleague_id"])
                    if c:
                        c.status = "training"
                progress_label.text = "Training started... (simulated)"
                ui.notify("Training pipeline initiated!", type="positive")

                # Simulate training progress
                for stage in ["Stage 2: Domain Specialization", "Stage 3: General Instruct", "Stage 4: Personalization"]:
                    progress_label.text = f"Running {stage}..."
                    await asyncio.sleep(2)

                with get_session(engine) as session:
                    c = session.get(Colleague, state["colleague_id"])
                    if c:
                        c.status = "ready"
                progress_label.text = "Training complete! Your AI Colleague is ready."
                ui.notify("Training complete!", type="positive")

            ui.button(
                "Start Training",
                on_click=start_training,
                icon="rocket_launch",
            ).classes("mt-4").props("color=positive size=lg")

            with ui.stepper_navigation():
                ui.button(
                    "Back to Home",
                    on_click=lambda: ui.navigate.to("/"),
                ).props("flat")
                ui.button("Back", on_click=stepper.previous).props("flat")


# ============================================================
# Run
# ============================================================

ui.run(title="NotGPT — AI Colleague Studio", port=8080, reload=True)
