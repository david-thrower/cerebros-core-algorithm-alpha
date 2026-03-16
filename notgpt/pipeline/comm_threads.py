from __future__ import annotations
import re
from notgpt.storage.db import get_session
from notgpt.storage.models import ColleagueDocument, ColleagueSyntheticSample


def process(colleague_id: int, engine, generators, num_samples: int = 3):
    """Process communication threads into instruct samples."""
    with get_session(engine) as session:
        docs = (
            session.query(ColleagueDocument)
            .filter_by(colleague_id=colleague_id, category="communication")
            .all()
        )
        for doc in docs:
            if not doc.extracted_text or not doc.extracted_text.strip():
                doc.processing_status = "failed"
                continue

            doc.processing_status = "processing"
            session.flush()

            # Split into message blocks (simple heuristic: double newline separated)
            blocks = [b.strip() for b in re.split(r"\n{2,}", doc.extracted_text) if b.strip()]

            # Pair consecutive blocks as inbound/outbound
            for j in range(0, len(blocks) - 1, 2):
                inbound = blocks[j]
                outbound = blocks[j + 1] if j + 1 < len(blocks) else ""
                if not outbound:
                    continue

                for i in range(num_samples):
                    variant_prompt = generators.generate_prompt(outbound)
                    reasoning = generators.generate_reasoning(outbound, variant_prompt)

                    sample = ColleagueSyntheticSample(
                        colleague_id=colleague_id,
                        source_type="communication",
                        source_id=doc.id,
                        synthetic_prompt=variant_prompt,
                        synthetic_reasoning=reasoning,
                        synthetic_response=outbound,
                        prompt_style=f"comm_variant_{i}",
                    )
                    session.add(sample)

            doc.processing_status = "done"
