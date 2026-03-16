from __future__ import annotations
from notgpt.storage.db import get_session
from notgpt.storage.models import ColleagueDocument, ColleagueSyntheticSample


def process(colleague_id: int, engine, generators, num_samples: int = 3):
    """Process reference documents into pretraining-style samples.

    Unlike other pipelines, references produce text variants (not instruct format).
    The synthetic_response contains the variant text, synthetic_prompt is minimal.
    """
    with get_session(engine) as session:
        docs = (
            session.query(ColleagueDocument)
            .filter_by(colleague_id=colleague_id, category="reference")
            .all()
        )
        for doc in docs:
            if not doc.extracted_text or not doc.extracted_text.strip():
                doc.processing_status = "failed"
                continue

            doc.processing_status = "processing"
            session.flush()

            # For references, we generate text variants for pretraining
            # No instruct format — just varied versions of the reference content
            for i in range(num_samples):
                # Use the generator to create a short topic summary as "prompt"
                topic = generators.generate_prompt(doc.extracted_text)
                # The "response" is the original reference text (preserved)
                # Reasoning is a brief note about the reference
                reasoning = generators.generate_reasoning(doc.extracted_text, topic)

                sample = ColleagueSyntheticSample(
                    colleague_id=colleague_id,
                    source_type="reference",
                    source_id=doc.id,
                    synthetic_prompt=topic,
                    synthetic_reasoning=reasoning,
                    synthetic_response=doc.extracted_text,
                    prompt_style=f"reference_{i}",
                )
                session.add(sample)

            doc.processing_status = "done"
