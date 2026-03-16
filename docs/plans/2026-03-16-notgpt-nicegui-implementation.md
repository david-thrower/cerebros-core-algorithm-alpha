# NotGPT NiceGUI Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone 7-step NiceGUI dashboard for the Cerebros NotGPT AI Colleague onboarding workflow — upload data, run preprocessing pipelines, review synthetic samples, trigger training.

**Architecture:** Single-process Python app using NiceGUI's `ui.stepper` for the 7-step wizard. SQLite/SQLAlchemy for state. Pipeline modules imported directly. Background tasks via `asyncio` + NiceGUI's `background_tasks.create()`. MLflow optional.

**Tech Stack:** NiceGUI, SQLAlchemy, pdfplumber, python-docx, transformers (optional for LLM mode), mlflow (optional), pendulum

---

### Task 1: Requirements and project scaffolding

**Files:**
- Create: `notgpt/notgpt-requirements.txt`
- Create: `notgpt/__init__.py`
- Create: `notgpt/pipeline/__init__.py`
- Create: `notgpt/storage/__init__.py`
- Create: `notgpt/components/__init__.py`

**Step 1: Create requirements file**

```
nicegui>=2.20.0
sqlalchemy>=2.0
pdfplumber>=0.11
python-docx>=1.1
pendulum>=3.0
aiofiles>=24.0
```

**Step 2: Create empty __init__.py files**

Touch `notgpt/__init__.py`, `notgpt/pipeline/__init__.py`, `notgpt/storage/__init__.py`, `notgpt/components/__init__.py`.

**Step 3: Install deps**

Run: `pip install -r notgpt/notgpt-requirements.txt`

**Step 4: Commit**

```bash
git add notgpt/
git commit -m "feat: scaffold notgpt NiceGUI app structure"
```

---

### Task 2: SQLAlchemy storage layer

**Files:**
- Create: `notgpt/storage/models.py`
- Create: `notgpt/storage/db.py`
- Create: `notgpt/tests/test_storage.py`

**Step 1: Write the failing test**

```python
# notgpt/tests/test_storage.py
import pytest
from notgpt.storage.db import get_engine, get_session, init_db
from notgpt.storage.models import Colleague, ColleagueDocument, ColleagueQAPair, ColleagueSyntheticSample

def test_create_colleague():
    engine = get_engine(":memory:")
    init_db(engine)
    with get_session(engine) as session:
        c = Colleague(name="Test Assistant", description="Test")
        session.add(c)
        session.commit()
        assert c.id is not None
        assert c.status == "draft"

def test_create_document():
    engine = get_engine(":memory:")
    init_db(engine)
    with get_session(engine) as session:
        c = Colleague(name="Test")
        session.add(c)
        session.flush()
        doc = ColleagueDocument(
            colleague_id=c.id,
            category="work_product",
            original_filename="report.pdf",
            extracted_text="Some text",
        )
        session.add(doc)
        session.commit()
        assert doc.id is not None
        assert doc.processing_status == "pending"

def test_create_qa_pair():
    engine = get_engine(":memory:")
    init_db(engine)
    with get_session(engine) as session:
        c = Colleague(name="Test")
        session.add(c)
        session.flush()
        qa = ColleagueQAPair(
            colleague_id=c.id,
            prompt="Why is the sky blue?",
            response="Rayleigh scattering.",
        )
        session.add(qa)
        session.commit()
        assert qa.id is not None
        assert qa.reasoning is None

def test_create_synthetic_sample():
    engine = get_engine(":memory:")
    init_db(engine)
    with get_session(engine) as session:
        c = Colleague(name="Test")
        session.add(c)
        session.flush()
        s = ColleagueSyntheticSample(
            colleague_id=c.id,
            source_type="work_product",
            synthetic_prompt="Write a report",
            synthetic_reasoning="<think>steps</think>",
            synthetic_response="The report content",
            prompt_style="llm_reverse_engineered",
        )
        session.add(s)
        session.commit()
        assert s.id is not None
        assert s.approved is False
```

**Step 2: Run test to verify it fails**

Run: `cd /home/mo/DEV/cerebros-ui && python -m pytest notgpt/tests/test_storage.py -v`
Expected: FAIL (modules don't exist)

**Step 3: Write models.py**

```python
# notgpt/storage/models.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass

class Colleague(Base):
    __tablename__ = "colleagues"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    status = Column(String(50), default="draft")  # draft|processing|reviewing|training|ready|failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    documents = relationship("ColleagueDocument", back_populates="colleague", cascade="all, delete-orphan")
    qa_pairs = relationship("ColleagueQAPair", back_populates="colleague", cascade="all, delete-orphan")
    synthetic_samples = relationship("ColleagueSyntheticSample", back_populates="colleague", cascade="all, delete-orphan")

class ColleagueDocument(Base):
    __tablename__ = "colleague_documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    colleague_id = Column(Integer, ForeignKey("colleagues.id"), nullable=False)
    category = Column(String(50), nullable=False)  # work_product|communication|reference
    original_filename = Column(String(512), nullable=False)
    extracted_text = Column(Text, default="")
    processing_status = Column(String(50), default="pending")  # pending|processing|done|failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    colleague = relationship("Colleague", back_populates="documents")

class ColleagueQAPair(Base):
    __tablename__ = "colleague_qa_pairs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    colleague_id = Column(Integer, ForeignKey("colleagues.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    colleague = relationship("Colleague", back_populates="qa_pairs")

class ColleagueSyntheticSample(Base):
    __tablename__ = "colleague_synthetic_samples"
    id = Column(Integer, primary_key=True, autoincrement=True)
    colleague_id = Column(Integer, ForeignKey("colleagues.id"), nullable=False)
    source_type = Column(String(50), nullable=False)  # work_product|qa|communication|reference
    source_id = Column(Integer, nullable=True)
    synthetic_prompt = Column(Text, nullable=False)
    synthetic_reasoning = Column(Text, default="")
    synthetic_response = Column(Text, default="")
    prompt_style = Column(String(100), default="")
    approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    colleague = relationship("Colleague", back_populates="synthetic_samples")
```

**Step 4: Write db.py**

```python
# notgpt/storage/db.py
from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from notgpt.storage.models import Base

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "notgpt.db"

def get_engine(db_path: str | None = None):
    if db_path is None:
        db_path = os.getenv("NOTGPT_DB_PATH", str(_DEFAULT_DB_PATH))
    if db_path == ":memory:":
        url = "sqlite:///:memory:"
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    return create_engine(url, echo=False)

def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine

@contextmanager
def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

**Step 5: Run tests**

Run: `cd /home/mo/DEV/cerebros-ui && python -m pytest notgpt/tests/test_storage.py -v`
Expected: All 4 PASS

**Step 6: Commit**

```bash
git add notgpt/storage/ notgpt/tests/
git commit -m "feat: add SQLAlchemy storage layer for colleague onboarding"
```

---

### Task 3: Text extraction module

**Files:**
- Create: `notgpt/pipeline/text_extract.py`
- Create: `notgpt/tests/test_text_extract.py`

**Step 1: Write the failing test**

```python
# notgpt/tests/test_text_extract.py
import pytest
from notgpt.pipeline.text_extract import extract_text

def test_extract_plain_text(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world. This is a test document.")
    text = extract_text(str(f))
    assert "Hello world" in text

def test_extract_returns_empty_for_missing():
    text = extract_text("/nonexistent/file.txt")
    assert text == ""

def test_extract_markdown(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Title\n\nSome content here.")
    text = extract_text(str(f))
    assert "Title" in text
    assert "content" in text
```

**Step 2: Run to verify failure**

Run: `cd /home/mo/DEV/cerebros-ui && python -m pytest notgpt/tests/test_text_extract.py -v`

**Step 3: Implement text_extract.py**

```python
# notgpt/pipeline/text_extract.py
from __future__ import annotations
import os
from pathlib import Path

def extract_text(file_path: str) -> str:
    """Extract text from PDF, DOCX, or plain text files."""
    path = Path(file_path)
    if not path.exists():
        return ""

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        # Plain text, markdown, csv, etc.
        return _extract_plain(path)


def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception:
        return ""


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""


def _extract_plain(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
```

**Step 4: Run tests**

Run: `cd /home/mo/DEV/cerebros-ui && python -m pytest notgpt/tests/test_text_extract.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add notgpt/pipeline/text_extract.py notgpt/tests/test_text_extract.py
git commit -m "feat: add text extraction for PDF, DOCX, and plain text"
```

---

### Task 4: Generator module (shared across pipelines)

**Files:**
- Create: `notgpt/pipeline/generators.py`

**Step 1: Copy and adapt generators from the existing pipeline**

Extract the `Generators` class, `heuristic_reverse_engineer_prompt`, `heuristic_reasoning_from_response`, `llm_reverse_engineer_prompt`, `llm_reverse_engineer_reasoning`, `build_text_generation_pipeline`, `draft_single_text`, and `clean_generated_text` from `/home/mo/DEV/prototype_work_product_data_engineering_pipeline_mlflow.py` into `notgpt/pipeline/generators.py`.

This is a direct copy of the working code — the LLM system prompts, tokenizer-based fallback, and the `Generators` class with `generate_prompt()` and `generate_reasoning()`.

**Step 2: Verify import works**

Run: `cd /home/mo/DEV/cerebros-ui && python -c "from notgpt.pipeline.generators import Generators; g = Generators(use_llm=False, target_seq_len=500); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add notgpt/pipeline/generators.py
git commit -m "feat: add shared generator module (LLM + tokenizer fallback)"
```

---

### Task 5: Pipeline modules (#1-#4)

**Files:**
- Create: `notgpt/pipeline/work_products.py`
- Create: `notgpt/pipeline/qa_upsampling.py`
- Create: `notgpt/pipeline/comm_threads.py`
- Create: `notgpt/pipeline/references.py`

Each pipeline module exposes `process(colleague_id: int, engine, generators: Generators)` that:
1. Reads source data from SQLite (documents or QA pairs for this colleague)
2. Runs generators
3. Writes `ColleagueSyntheticSample` rows back to SQLite

**Step 1: Implement work_products.py**

Reads `ColleagueDocument` where category='work_product', runs `generators.generate_prompt()` and `generators.generate_reasoning()` on each document's extracted_text, writes `ColleagueSyntheticSample` rows.

**Step 2: Implement qa_upsampling.py**

Reads `ColleagueQAPair`, runs generators to produce synthetic variants of each prompt/reasoning/response triplet.

**Step 3: Implement comm_threads.py**

Reads `ColleagueDocument` where category='communication', parses into inbound/outbound turns, generates synthetic instruct samples.

**Step 4: Implement references.py**

Reads `ColleagueDocument` where category='reference', generates pretraining-style summaries (no instruct format — just text variants).

**Step 5: Test each pipeline with deterministic generators**

Run: `cd /home/mo/DEV/cerebros-ui && python -c "from notgpt.pipeline.work_products import process; print('import OK')"`

**Step 6: Commit**

```bash
git add notgpt/pipeline/
git commit -m "feat: add 4 preprocessing pipeline modules"
```

---

### Task 6: NiceGUI app — main entry + splash

**Files:**
- Create: `notgpt/app.py`

**Step 1: Create the main app with splash page (Step 0)**

```python
# notgpt/app.py
from __future__ import annotations
from nicegui import ui, app
from notgpt.storage.db import get_engine, init_db, get_session
from notgpt.storage.models import Colleague

engine = None

def startup():
    global engine
    engine = init_db()

app.on_startup(startup)

@ui.page("/")
def splash_page():
    ui.label("NotGPT — AI Colleague Studio").classes("text-3xl font-bold mb-4")

    with get_session(engine) as session:
        colleagues = session.query(Colleague).order_by(Colleague.created_at.desc()).all()
        colleague_data = [{"id": c.id, "name": c.name, "status": c.status} for c in colleagues]

    if colleague_data:
        ui.table(
            columns=[
                {"name": "name", "label": "Name", "field": "name"},
                {"name": "status", "label": "Status", "field": "status"},
            ],
            rows=colleague_data,
        ).classes("w-full")
    else:
        ui.label("No colleagues yet. Create your first one!").classes("text-gray-500")

    ui.button("Create New Colleague", on_click=lambda: ui.navigate.to("/new")).classes("mt-4")

ui.run(title="NotGPT", port=8080, reload=True)
```

**Step 2: Run and verify splash page loads**

Run: `cd /home/mo/DEV/cerebros-ui && python notgpt/app.py`
Expected: Browser opens at localhost:8080, shows empty colleague list + create button.

**Step 3: Commit**

```bash
git add notgpt/app.py
git commit -m "feat: add NiceGUI splash page with colleague list"
```

---

### Task 7: NiceGUI wizard — 7-step stepper

**Files:**
- Modify: `notgpt/app.py` — add `/new` page with stepper

**Step 1: Add the /new page with ui.stepper containing all 7 steps**

Each step uses `ui.step()`. Navigation via `stepper.next()` / `stepper.previous()`. State stored in a dict bound to the page.

- **Step 1 (Metadata)**: Name + description inputs. On "Next": creates Colleague in SQLite.
- **Step 2 (Work Products)**: `ui.upload(multiple=True)` for files. On upload: extract text, store as ColleagueDocument. On "Next": trigger pipeline #1 in background.
- **Step 3 (Q&A)**: Table with add/edit/delete rows. Stores as ColleagueQAPair.
- **Step 4 (Comm Threads)**: Platform selector + identity input + file upload.
- **Step 5 (References)**: File upload for manuals/SOPs.
- **Step 6 (Review)**: Table of all ColleagueSyntheticSample rows. Edit/delete. "Approve" button.
- **Step 7 (Training)**: Summary + "Start Training" button + progress label with timer polling.

**Step 2: Each step implementation follows the NiceGUI stepper pattern**

```python
@ui.page("/new")
def wizard_page():
    state = {"colleague_id": None}

    with ui.stepper().classes("w-full") as stepper:
        # Step 1: Metadata
        with ui.step("Name Your Colleague"):
            name_input = ui.input("Name", placeholder="e.g. Sales Assistant")
            desc_input = ui.textarea("Description (optional)")
            with ui.stepper_navigation():
                ui.button("Next", on_click=lambda: _save_metadata_and_next(
                    state, name_input, desc_input, stepper))

        # Step 2: Work Products
        with ui.step("Upload Work Products"):
            ui.label("Upload examples of your completed deliverables")
            upload = ui.upload(
                multiple=True,
                on_upload=lambda e: _handle_upload(e, state, "work_product"),
            ).props('accept=".pdf,.docx,.doc,.txt,.md"')
            with ui.stepper_navigation():
                ui.button("Next", on_click=stepper.next)
                ui.button("Back", on_click=stepper.previous).props("flat")

        # ... Steps 3-7 follow same pattern
```

**Step 3: Wire background pipeline execution**

When navigating away from upload steps, trigger the corresponding pipeline in a background task:

```python
from nicegui import background_tasks

async def _run_pipeline(colleague_id, pipeline_fn):
    from notgpt.pipeline.generators import Generators
    generators = Generators(use_llm=False, target_seq_len=500)
    pipeline_fn(colleague_id, engine, generators)

# Called on "Next" from Step 2:
background_tasks.create(_run_pipeline(state["colleague_id"], work_products.process))
```

**Step 4: Test the full wizard flow manually**

Run: `cd /home/mo/DEV/cerebros-ui && python notgpt/app.py`
Navigate: localhost:8080 → "Create New" → fill name → next → upload file → next → ... → step 7

**Step 5: Commit**

```bash
git add notgpt/
git commit -m "feat: add 7-step NiceGUI wizard for colleague onboarding"
```

---

### Task 8: Review table with edit/delete/approve

**Files:**
- Modify: `notgpt/app.py` — Step 6 review implementation

**Step 1: Implement Step 6 with editable table**

Query all `ColleagueSyntheticSample` for this colleague, display in a table with:
- Columns: source_type, synthetic_prompt (truncated), synthetic_reasoning (truncated), approved
- Row actions: Edit (opens dialog), Delete, Toggle approve
- "Approve All" button
- Sample count display

**Step 2: Test review flow**

Create a colleague, upload a file, wait for pipeline, navigate to Step 6, verify samples appear.

**Step 3: Commit**

```bash
git add notgpt/
git commit -m "feat: add synthetic sample review table with edit/delete/approve"
```

---

### Task 9: Training step with progress

**Files:**
- Modify: `notgpt/app.py` — Step 7 training implementation

**Step 1: Implement Step 7**

- Summary: counts of approved samples by source type
- "Start Training" button: updates colleague status to "training", enqueues training job
- Progress display: `ui.timer(2.0, callback)` polls colleague status from SQLite
- On completion: show "Ready" status with link back to splash

For MVP, training is a placeholder that simulates progress (since actual GPU training is separate). The pipeline infrastructure (MLflow logging, dataset assembly) is real.

**Step 2: Commit**

```bash
git add notgpt/
git commit -m "feat: add training confirmation and progress display"
```

---

### Task 10: End-to-end smoke test and polish

**Files:**
- Create: `notgpt/tests/test_e2e.py`

**Step 1: Write an end-to-end test**

```python
# notgpt/tests/test_e2e.py
from notgpt.storage.db import get_engine, init_db, get_session
from notgpt.storage.models import Colleague, ColleagueDocument, ColleagueSyntheticSample
from notgpt.pipeline.generators import Generators
from notgpt.pipeline import work_products

def test_full_pipeline_flow():
    engine = get_engine(":memory:")
    init_db(engine)
    generators = Generators(use_llm=False, target_seq_len=500)

    # Create colleague
    with get_session(engine) as session:
        c = Colleague(name="Test Assistant")
        session.add(c)
        session.flush()
        cid = c.id

        # Add a document
        doc = ColleagueDocument(
            colleague_id=cid,
            category="work_product",
            original_filename="test.txt",
            extracted_text="The quarterly report shows revenue of $4.2M with 18% QoQ growth.",
        )
        session.add(doc)

    # Run pipeline
    work_products.process(cid, engine, generators)

    # Verify synthetic samples were created
    with get_session(engine) as session:
        samples = session.query(ColleagueSyntheticSample).filter_by(colleague_id=cid).all()
        assert len(samples) > 0
        for s in samples:
            assert s.synthetic_prompt  # Not empty
            assert s.source_type == "work_product"
            assert s.approved is False
```

**Step 2: Run**

Run: `cd /home/mo/DEV/cerebros-ui && python -m pytest notgpt/tests/test_e2e.py -v`

**Step 3: Final commit**

```bash
git add notgpt/
git commit -m "feat: add e2e test and polish NotGPT NiceGUI dashboard"
```

---

## Verification Checklist

1. `pip install -r notgpt/notgpt-requirements.txt` — installs without errors
2. `python notgpt/app.py` — opens browser at :8080
3. Create colleague → upload .txt file → see it listed → next
4. Add Q&A pairs → next
5. Review synthetic samples in Step 6
6. Step 7 shows summary
7. `python -m pytest notgpt/tests/ -v` — all tests pass
8. SQLite DB at `notgpt/data/notgpt.db` contains data
