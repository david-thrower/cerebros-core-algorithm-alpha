from __future__ import annotations
from notgpt.storage.db import get_session
from notgpt.storage.models import ColleagueQAPair, ColleagueSyntheticSample


def process(colleague_id: int, engine, generators, num_samples: int = 3):
    """Upsample Q&A pairs into synthetic variants."""
    with get_session(engine) as session:
        pairs = (
            session.query(ColleagueQAPair)
            .filter_by(colleague_id=colleague_id)
            .all()
        )
        for qa in pairs:
            for i in range(num_samples):
                # Generate variant prompt from the original response
                variant_prompt = generators.generate_prompt(qa.response)
                # Generate reasoning for this variant
                reasoning = generators.generate_reasoning(qa.response, variant_prompt)

                sample = ColleagueSyntheticSample(
                    colleague_id=colleague_id,
                    source_type="qa",
                    source_id=qa.id,
                    synthetic_prompt=variant_prompt,
                    synthetic_reasoning=reasoning,
                    synthetic_response=qa.response,
                    prompt_style=f"qa_variant_{i}",
                )
                session.add(sample)
