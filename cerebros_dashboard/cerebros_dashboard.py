"""Cerebros NotGPT — AI Colleague Studio."""

import reflex as rx

from cerebros_dashboard.state import AppState, WizardState


# ─────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────

def header() -> rx.Component:
    return rx.hstack(
        rx.image(src="/cerebros-logo.png", height="48px"),
        rx.text(
            "NotGPT",
            class_name="gradient-text text-2xl font-bold",
        ),
        rx.spacer(),
        rx.link(
            rx.text("Home", class_name="text-sm text-[#718096] hover:text-[#1A202C]"),
            href="/",
        ),
        class_name="w-full px-8 py-4 border-b border-[#E2E8F0] items-center",
    )


def step_indicator(current: rx.Var[int]) -> rx.Component:
    steps = [
        "Name", "Work Products", "Q&A",
        "Comms", "References", "Review", "Train",
    ]

    def step_dot(label: str, idx: int) -> rx.Component:
        return rx.vstack(
            rx.box(
                rx.text(str(idx + 1), class_name="text-xs font-bold"),
                class_name=rx.cond(
                    current > idx,
                    "step-done w-8 h-8 rounded-full flex items-center justify-center",
                    rx.cond(
                        current == idx,
                        "step-active w-8 h-8 rounded-full flex items-center justify-center",
                        "step-pending w-8 h-8 rounded-full flex items-center justify-center",
                    ),
                ),
            ),
            rx.text(
                label,
                class_name="text-xs text-[#718096] mt-1",
            ),
            align="center",
            spacing="1",
        )

    return rx.hstack(
        *[step_dot(label, i) for i, label in enumerate(steps)],
        class_name="w-full justify-between px-4 py-6 max-w-3xl mx-auto",
    )


def upload_zone(upload_id: str, handler, label: str) -> rx.Component:
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.icon("upload", size=32, class_name="text-[#16CEEB] mx-auto"),
                rx.text(label, class_name="text-[#718096] text-center"),
                rx.text(
                    "Drag and drop files here or click to select",
                    class_name="text-sm text-[#A0AEC0] text-center",
                ),
                align="center",
                spacing="2",
            ),
            id=upload_id,
            class_name="upload-zone w-full",
            multiple=True,
        ),
        rx.hstack(
            rx.foreach(
                rx.selected_files(upload_id),
                lambda f: rx.text(f, class_name="text-xs text-[#718096]"),
            ),
            wrap="wrap",
        ),
        rx.button(
            "Upload Files",
            on_click=handler(rx.upload_files(upload_id=upload_id)),
            class_name="btn-secondary",
        ),
        spacing="3",
        class_name="w-full",
    )


# ─────────────────────────────────────────────
# Wizard Steps
# ─────────────────────────────────────────────

def step_metadata() -> rx.Component:
    return rx.vstack(
        rx.text(
            "What should we call your AI Colleague?",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Give it a name and optionally describe what it will help with.",
            class_name="text-[#718096]",
        ),
        rx.input(
            placeholder="e.g. Sales Assistant, Legal Analyst, Project Manager",
            value=WizardState.colleague_name,
            on_change=WizardState.set_colleague_name,
            class_name="w-full max-w-md p-3 border border-[#E2E8F0] rounded-lg text-lg",
        ),
        rx.text_area(
            placeholder="Description (optional) — What will this colleague help with?",
            value=WizardState.colleague_desc,
            on_change=WizardState.set_colleague_desc,
            class_name="w-full max-w-md p-3 border border-[#E2E8F0] rounded-lg",
        ),
        rx.button(
            "Next",
            on_click=WizardState.save_colleague,
            class_name="btn-primary",
        ),
        spacing="4",
        class_name="card w-full max-w-2xl",
    )


def step_work_products() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Upload Work Products",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Upload examples of your completed deliverables — reports, invoices, briefs, "
            "documents. The more examples, the better your AI Colleague will understand your style.",
            class_name="text-[#718096]",
        ),
        upload_zone(
            "wp_upload",
            WizardState.handle_wp_upload,
            "PDF, DOCX, TXT, MD files",
        ),
        rx.cond(
            WizardState.wp_files.length() > 0,
            rx.vstack(
                rx.text(
                    rx.cond(
                        WizardState.wp_processing,
                        "Processing files...",
                        "Files uploaded:",
                    ),
                    class_name="text-sm font-medium text-[#1A202C]",
                ),
                rx.foreach(
                    WizardState.wp_files,
                    lambda f: rx.hstack(
                        rx.icon("file-text", size=16, class_name="text-[#16CEEB]"),
                        rx.text(f["name"], class_name="text-sm"),
                        rx.text(
                            f["chars"].to(str) + " chars",
                            class_name="text-xs text-[#A0AEC0]",
                        ),
                    ),
                ),
                spacing="2",
            ),
        ),
        rx.hstack(
            rx.button("Back", on_click=WizardState.prev_step, class_name="btn-ghost"),
            rx.button(
                rx.cond(WizardState.wp_processing, "Processing...", "Next"),
                on_click=WizardState.next_from_wp,
                class_name="btn-primary",
                disabled=WizardState.wp_processing,
            ),
            spacing="3",
        ),
        spacing="4",
        class_name="card w-full max-w-2xl",
    )


def step_qa() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Questions & Answers",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Add examples of questions you get asked and how you answer them. "
            "Think of it as explaining your job to an intern.",
            class_name="text-[#718096]",
        ),
        rx.vstack(
            rx.input(
                placeholder="Question or prompt...",
                value=WizardState.qa_prompt,
                on_change=WizardState.set_qa_prompt,
                class_name="w-full p-3 border border-[#E2E8F0] rounded-lg",
            ),
            rx.input(
                placeholder="Reasoning / thought process (optional)",
                value=WizardState.qa_reasoning,
                on_change=WizardState.set_qa_reasoning,
                class_name="w-full p-3 border border-[#E2E8F0] rounded-lg",
            ),
            rx.text_area(
                placeholder="Your answer / response...",
                value=WizardState.qa_response,
                on_change=WizardState.set_qa_response,
                class_name="w-full p-3 border border-[#E2E8F0] rounded-lg",
            ),
            rx.button(
                "Add Q&A Pair",
                on_click=WizardState.add_qa_pair,
                class_name="btn-secondary",
            ),
            spacing="2",
            class_name="w-full",
        ),
        rx.cond(
            WizardState.qa_pairs.length() > 0,
            rx.vstack(
                rx.text(
                    WizardState.qa_pairs.length().to(str) + " pairs added",
                    class_name="text-sm font-medium",
                ),
                rx.foreach(
                    WizardState.qa_pairs,
                    lambda qa: rx.hstack(
                        rx.icon("message-circle", size=16, class_name="text-[#D04C90]"),
                        rx.text(qa["prompt"], class_name="text-sm flex-1"),
                        spacing="2",
                    ),
                ),
                spacing="2",
            ),
        ),
        rx.hstack(
            rx.button("Back", on_click=WizardState.prev_step, class_name="btn-ghost"),
            rx.button("Next", on_click=WizardState.next_from_qa, class_name="btn-primary"),
            spacing="3",
        ),
        spacing="4",
        class_name="card w-full max-w-2xl",
    )


def step_comms() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Communication Threads",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Upload examples of your email threads, Slack messages, or Discord conversations.",
            class_name="text-[#718096]",
        ),
        rx.hstack(
            rx.select(
                ["Email", "Slack", "Discord", "Other"],
                value=WizardState.comm_platform,
                on_change=WizardState.set_comm_platform,
                class_name="p-2 border border-[#E2E8F0] rounded-lg",
            ),
            rx.input(
                placeholder="Your email or handle",
                value=WizardState.comm_identity,
                on_change=WizardState.set_comm_identity,
                class_name="flex-1 p-3 border border-[#E2E8F0] rounded-lg",
            ),
            spacing="3",
            class_name="w-full",
        ),
        upload_zone(
            "comm_upload",
            WizardState.handle_comm_upload,
            "Thread exports (.txt, .md, .csv, .eml)",
        ),
        rx.hstack(
            rx.button("Back", on_click=WizardState.prev_step, class_name="btn-ghost"),
            rx.button("Next", on_click=WizardState.next_from_comms, class_name="btn-primary"),
            spacing="3",
        ),
        spacing="4",
        class_name="card w-full max-w-2xl",
    )


def step_references() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Internal References",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Upload manuals, SOPs, policies, law books — any reference materials "
            "you use to do your job.",
            class_name="text-[#718096]",
        ),
        upload_zone(
            "ref_upload",
            WizardState.handle_ref_upload,
            "PDF, DOCX, TXT, MD files",
        ),
        rx.hstack(
            rx.button("Back", on_click=WizardState.prev_step, class_name="btn-ghost"),
            rx.button("Next", on_click=WizardState.next_from_refs, class_name="btn-primary"),
            spacing="3",
        ),
        spacing="4",
        class_name="card w-full max-w-2xl",
    )


def step_review() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Review Synthetic Data",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Review the training samples generated from your data. "
            "Delete any that don't look right, then approve.",
            class_name="text-[#718096]",
        ),
        rx.button(
            "Load Samples",
            on_click=WizardState.load_samples,
            class_name="btn-secondary",
        ),
        rx.cond(
            WizardState.synthetic_samples.length() > 0,
            rx.vstack(
                rx.hstack(
                    rx.text(
                        WizardState.synthetic_samples.length().to(str) + " samples",
                        class_name="text-lg font-semibold",
                    ),
                    rx.button(
                        "Approve All",
                        on_click=WizardState.approve_all,
                        class_name="btn-primary",
                    ),
                    spacing="4",
                    align="center",
                ),
                rx.foreach(
                    WizardState.synthetic_samples,
                    lambda s: rx.hstack(
                        rx.badge(s["source"], class_name="text-xs"),
                        rx.text(s["prompt"], class_name="text-sm flex-1"),
                        rx.cond(
                            s["approved"],
                            rx.icon(
                                "check-circle",
                                size=16,
                                class_name="text-green-500",
                            ),
                            rx.icon(
                                "circle",
                                size=16,
                                class_name="text-[#A0AEC0]",
                            ),
                        ),
                        rx.button(
                            rx.icon("trash-2", size=14),
                            on_click=WizardState.delete_sample(s["id"]),
                            class_name="text-red-400 hover:text-red-600",
                            variant="ghost",
                            size="1",
                        ),
                        class_name="w-full p-3 border-b border-[#E2E8F0] items-center",
                        spacing="3",
                    ),
                ),
                spacing="2",
                class_name="w-full max-h-96 overflow-y-auto",
            ),
            rx.text(
                "No samples yet. Go back and upload data first.",
                class_name="text-[#A0AEC0] italic",
            ),
        ),
        rx.hstack(
            rx.button("Back", on_click=WizardState.prev_step, class_name="btn-ghost"),
            rx.button("Next", on_click=WizardState.next_step, class_name="btn-primary"),
            spacing="3",
        ),
        spacing="4",
        class_name="card w-full max-w-3xl",
    )


def step_training() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Train Your AI Colleague",
            class_name="text-xl font-semibold text-[#1A202C]",
        ),
        rx.text(
            "Your data is ready. Start the 3-stage training pipeline.",
            class_name="text-[#718096]",
        ),
        rx.vstack(
            rx.hstack(
                rx.text("Stage 2:", class_name="font-semibold w-24"),
                rx.text("Domain Specialization", class_name="text-[#718096]"),
            ),
            rx.hstack(
                rx.text("Stage 3:", class_name="font-semibold w-24"),
                rx.text("General Instruction Following", class_name="text-[#718096]"),
            ),
            rx.hstack(
                rx.text("Stage 4:", class_name="font-semibold w-24"),
                rx.text("Your Personal Style & Knowledge", class_name="text-[#718096]"),
            ),
            spacing="2",
            class_name="w-full p-4 bg-[#F7FAFC] rounded-lg",
        ),
        rx.cond(
            WizardState.training_status == "idle",
            rx.button(
                "Start Training",
                on_click=WizardState.start_training,
                class_name="btn-primary text-lg px-8 py-3",
            ),
            rx.vstack(
                rx.text(
                    WizardState.training_stage,
                    class_name="text-lg font-semibold text-[#16CEEB]",
                ),
                rx.progress(value=WizardState.training_progress, class_name="w-full"),
                rx.text(
                    WizardState.training_progress.to(str) + "%",
                    class_name="text-sm text-[#718096]",
                ),
                rx.cond(
                    WizardState.training_status == "complete",
                    rx.vstack(
                        rx.icon(
                            "check-circle",
                            size=48,
                            class_name="text-green-500 mx-auto",
                        ),
                        rx.text(
                            "Your AI Colleague is ready!",
                            class_name="text-xl font-bold text-green-600 text-center",
                        ),
                        rx.link(
                            rx.button("Back to Home", class_name="btn-primary"),
                            href="/",
                        ),
                        spacing="3",
                        align="center",
                    ),
                ),
                spacing="3",
                class_name="w-full",
            ),
        ),
        rx.cond(
            WizardState.training_status == "idle",
            rx.button("Back", on_click=WizardState.prev_step, class_name="btn-ghost"),
        ),
        spacing="4",
        class_name="card w-full max-w-2xl",
    )


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

def splash() -> rx.Component:
    return rx.vstack(
        header(),
        rx.vstack(
            rx.vstack(
                rx.text(
                    "Custom AI Assistants in Hours, Not Months",
                    class_name="gradient-text text-4xl font-bold text-center",
                ),
                rx.text(
                    "Upload your data. Review the training samples. "
                    "Get a fully personalized AI Colleague.",
                    class_name="text-[#718096] text-center text-lg max-w-xl mx-auto",
                ),
                spacing="3",
                class_name="py-12",
            ),
            rx.cond(
                AppState.colleagues.length() > 0,
                rx.vstack(
                    rx.text("Your Colleagues", class_name="text-xl font-semibold"),
                    rx.foreach(
                        AppState.colleagues,
                        lambda c: rx.hstack(
                            rx.icon("brain", size=20, class_name="text-[#D04C90]"),
                            rx.text(c["name"], class_name="font-medium"),
                            rx.spacer(),
                            rx.badge(c["status"]),
                            class_name="w-full p-4 card items-center",
                            spacing="3",
                        ),
                    ),
                    spacing="3",
                    class_name="w-full max-w-xl",
                ),
            ),
            rx.link(
                rx.button(
                    "Create New Colleague",
                    class_name="btn-primary text-lg px-8 py-3 mt-6",
                ),
                href="/new",
            ),
            align="center",
            class_name="px-8 py-8 max-w-4xl mx-auto",
        ),
        spacing="0",
        class_name="min-h-screen bg-white",
        on_mount=AppState.load_colleagues,
    )


def wizard() -> rx.Component:
    return rx.vstack(
        header(),
        step_indicator(WizardState.step),
        rx.box(
            rx.match(
                WizardState.step,
                (0, step_metadata()),
                (1, step_work_products()),
                (2, step_qa()),
                (3, step_comms()),
                (4, step_references()),
                (5, step_review()),
                (6, step_training()),
                step_metadata(),
            ),
            class_name="flex justify-center px-8 py-6",
        ),
        spacing="0",
        class_name="min-h-screen bg-[#F7FAFC]",
    )


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = rx.App(
    theme=rx.theme(appearance="light"),
    stylesheets=["/style.css"],
)
app.add_page(splash, route="/", title="NotGPT — AI Colleague Studio")
app.add_page(wizard, route="/new", title="Create AI Colleague — NotGPT")
