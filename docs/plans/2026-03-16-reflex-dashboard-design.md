# Cerebros NotGPT — Reflex Dashboard Design

**Date:** 2026-03-16
**Status:** Approved

## Context

Replace the NiceGUI prototype with a production-quality Reflex (Python-native React) dashboard matching cerebros.one branding. Same pipeline code, new UI.

## Architecture

Reflex app with Tailwind CSS, cerebros.one brand system, state-driven 7-step wizard. Reuses existing pipeline modules from `notgpt/pipeline/` and storage from `notgpt/storage/`.

## File Structure

```
cerebros_dashboard/
  cerebros_dashboard.py       # Main app + routing
  state.py                    # WizardState + ColleagueState
  pages/
    splash.py                 # Colleague list + create
    wizard.py                 # 7-step onboarding
  components/
    header.py                 # Logo + gradient nav
    upload_zone.py            # Drag-drop file upload
    qa_editor.py              # CRUD table for Q&A
    review_table.py           # Synthetic sample review/approve
    training_progress.py      # Stage 2-4 progress
    step_indicator.py         # Visual step progress bar
  assets/
    cerebros-logo.png
    style.css                 # Brand tokens
rxconfig.py                   # Tailwind config + Cerebros colors
```

## Brand System (from cerebros.one)

- Gradient: `linear-gradient(to right, #16CEEB, #D04C90)`
- Primary CTA: `#D04C90` (pink)
- Secondary: `#16CEEB` (cyan)
- Text: `#1A202C`
- Card bg: `#F7FAFC`
- Font: system stack

## State

```python
class WizardState(rx.State):
    step: int = 0
    colleague_id: int | None = None
    colleague_name: str = ""
    colleague_desc: str = ""
    uploaded_files: list[dict] = []
    qa_pairs: list[dict] = []
    comm_platform: str = "Email"
    comm_identity: str = ""
    synthetic_samples: list[dict] = []
    training_status: str = "idle"
    training_progress: int = 0
    processing: bool = False
```

## 7 Steps

0. Splash — colleague list, create button
1. Metadata — name + description
2. Work Products — file upload + pipeline #1
3. Q&A — CRUD editor + pipeline #2
4. Comm Threads — platform/identity + upload + pipeline #3
5. References — upload + pipeline #4
6. Review — sample table, edit/delete/approve
7. Training — summary, start, progress display
