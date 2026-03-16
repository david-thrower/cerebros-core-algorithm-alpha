from __future__ import annotations
from notgpt.storage.db import get_session
from notgpt.storage.models import ColleagueDocument, ColleagueSyntheticSample


def process(colleague_id: int, engine, generators, num_samples: int = 3):
    """Process work product documents into synthetic instruct samples."""
    with get_session(engine) as session:
        docs = (
            session.query(ColleagueDocument)
            .filter_by(colleague_id=colleague_id, category="work_product")
            .all()
        )
        for doc in docs:
            if not doc.extracted_text or not doc.extracted_text.strip():
                doc.processing_status = "failed"
                continue

            doc.processing_status = "processing"
            session.flush()

            for i in range(num_samples):
                prompt = generators.generate_prompt(doc.extracted_text)
                reasoning = generators.generate_reasoning(doc.extracted_text, prompt)

                sample = ColleagueSyntheticSample(
                    colleague_id=colleague_id,
                    source_type="work_product",
                    source_id=doc.id,
                    synthetic_prompt=prompt,
                    synthetic_reasoning=reasoning,
                    synthetic_response=doc.extracted_text,
                    prompt_style=f"sample_{i}",
                )
                session.add(sample)

            doc.processing_status = "done"
