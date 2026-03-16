# NotGPT NiceGUI Dashboard — Design Document

**Date:** 2026-03-16
**Status:** Approved for implementation

## Context

The Cerebros NotGPT platform needs a user-facing dashboard for the 7-step AI Colleague onboarding workflow. This dashboard must work standalone (no Thunderline dependency) so David can test the full upload-to-training flow independently. Later, the same pipeline code integrates into Thunderline's LiveView UI.

## Architecture

- **Framework:** NiceGUI (Python, no JS build, auto-reload)
- **Storage:** SQLite via SQLAlchemy (zero config)
- **Text extraction:** pdfplumber (PDF), python-docx (DOCX), plain text fallback
- **ML pipelines:** Direct Python import from pipeline modules
- **MLflow:** Optional — logs to MLflow if MLFLOW_TRACKING_URI is set, otherwise SQLite-only
- **LLM backend:** Configurable — HuggingFace pipeline (local Qwen) or REST endpoint

## File Structure

```
notgpt/
  app.py                      # NiceGUI entry point
  pipeline/
    __init__.py
    work_products.py           # Pipeline #1 — work product → synthetic instruct
    qa_upsampling.py           # Pipeline #2 — Q&A triplet upsampling
    comm_threads.py            # Pipeline #3 — communication thread → instruct
    references.py              # Pipeline #4 — internal refs → pretraining samples
    generators.py              # Shared: LLM + deterministic generator backends
    text_extract.py            # PDF/DOCX/TXT text extraction
  storage/
    __init__.py
    models.py                  # SQLAlchemy models
    db.py                      # Session management
  components/
    __init__.py
    wizard.py                  # 7-step wizard state machine
    upload.py                  # File upload component
    qa_editor.py               # CRUD table for Q&A pairs
    review_table.py            # Synthetic sample review/approval
    training_progress.py       # Training status display
  notgpt-requirements.txt      # UI-specific deps (nicegui, pdfplumber, etc.)
```

## 7-Step Wizard Flow

### Step 0: Splash
- List existing colleagues (name, status)
- "Create New Colleague" button

### Step 1: Metadata
- Name (required), description (optional)
- Creates Colleague record in SQLite

### Step 2: Upload Work Products
- Drag-drop file upload (PDF, DOCX, TXT)
- Text extraction via pdfplumber/python-docx
- Triggers pipeline #1 in background thread
- Shows per-file processing status

### Step 3: Q&A Pairs
- Inline CRUD table: prompt / reasoning (optional) / response
- Add/Edit/Delete
- Triggers pipeline #2 on "Next"

### Step 4: Communication Threads
- Platform selector (Email/Slack/Discord)
- Identity field (email/handle)
- File upload for thread exports
- Triggers pipeline #3

### Step 5: Internal References
- File upload for manuals/SOPs
- Triggers pipeline #4

### Step 6: Review Synthetic Data
- Table of all generated samples grouped by source
- Edit/Delete per row
- "Approve" button finalizes dataset
- Shows expansion preview (n samples → n approved)

### Step 7: Training Confirmation & Progress
- Summary of approved data
- "Start Training" button
- Progress display (polling or callback)
- On completion: status update

## Data Model (SQLAlchemy)

```
Colleague: id, name, description, status, created_at
ColleagueDocument: id, colleague_id, category, original_filename, extracted_text, processing_status
ColleagueQAPair: id, colleague_id, prompt, reasoning, response
ColleagueSyntheticSample: id, colleague_id, source_type, source_id, synthetic_prompt, synthetic_reasoning, synthetic_response, prompt_style, approved
```

## Pipeline Integration

Each pipeline module exposes a `process(colleague_id, db_session)` function that:
1. Reads source data from SQLite
2. Runs the generator (LLM or deterministic)
3. Writes synthetic samples back to SQLite
4. Optionally logs to MLflow

Generators from the existing `prototype_work_product_data_engineering_pipeline_mlflow.py` are extracted into `pipeline/generators.py` for reuse across all 4 pipelines.

## Verification

1. `pip install -r notgpt/notgpt-requirements.txt`
2. `python notgpt/app.py` — opens browser at localhost:8080
3. Create colleague → upload PDF → see extracted text → see synthetic samples
4. All 7 steps navigable
5. SQLite DB created automatically in `notgpt/data/notgpt.db`
