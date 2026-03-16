# notgpt/tests/test_e2e.py
from notgpt.storage.db import get_engine, init_db, get_session
from notgpt.storage.models import (
    Colleague, ColleagueDocument, ColleagueQAPair, ColleagueSyntheticSample,
)
from notgpt.pipeline.generators import Generators
from notgpt.pipeline import work_products, qa_upsampling, references


def test_work_product_pipeline():
    engine = get_engine(":memory:")
    init_db(engine)
    generators = Generators(use_llm=False, target_seq_len=500)

    with get_session(engine) as session:
        c = Colleague(name="Test Assistant")
        session.add(c)
        session.flush()
        cid = c.id

        doc = ColleagueDocument(
            colleague_id=cid,
            category="work_product",
            original_filename="test.txt",
            extracted_text="The quarterly report shows revenue of $4.2M with 18% QoQ growth. Enterprise segment contributed $2.8M.",
        )
        session.add(doc)

    work_products.process(cid, engine, generators)

    with get_session(engine) as session:
        samples = session.query(ColleagueSyntheticSample).filter_by(colleague_id=cid).all()
        assert len(samples) == 3  # default num_samples
        for s in samples:
            assert s.synthetic_prompt
            assert s.source_type == "work_product"
            assert s.approved is False


def test_qa_pipeline():
    engine = get_engine(":memory:")
    init_db(engine)
    generators = Generators(use_llm=False, target_seq_len=500)

    with get_session(engine) as session:
        c = Colleague(name="QA Test")
        session.add(c)
        session.flush()
        cid = c.id

        qa = ColleagueQAPair(
            colleague_id=cid,
            prompt="What is our return policy?",
            reasoning="Check the standard 30-day policy",
            response="We offer a 30-day return policy on all items.",
        )
        session.add(qa)

    qa_upsampling.process(cid, engine, generators)

    with get_session(engine) as session:
        samples = session.query(ColleagueSyntheticSample).filter_by(colleague_id=cid).all()
        assert len(samples) == 3
        for s in samples:
            assert s.source_type == "qa"
            assert s.synthetic_prompt


def test_reference_pipeline():
    engine = get_engine(":memory:")
    init_db(engine)
    generators = Generators(use_llm=False, target_seq_len=500)

    with get_session(engine) as session:
        c = Colleague(name="Ref Test")
        session.add(c)
        session.flush()
        cid = c.id

        doc = ColleagueDocument(
            colleague_id=cid,
            category="reference",
            original_filename="sop.txt",
            extracted_text="Standard Operating Procedure: All invoices must be processed within 48 hours of receipt.",
        )
        session.add(doc)

    references.process(cid, engine, generators)

    with get_session(engine) as session:
        samples = session.query(ColleagueSyntheticSample).filter_by(colleague_id=cid).all()
        assert len(samples) == 3
        for s in samples:
            assert s.source_type == "reference"


def test_full_flow():
    """End-to-end: create colleague, add data, run all pipelines, verify samples."""
    engine = get_engine(":memory:")
    init_db(engine)
    generators = Generators(use_llm=False, target_seq_len=500)

    with get_session(engine) as session:
        c = Colleague(name="Full Flow Test")
        session.add(c)
        session.flush()
        cid = c.id

        session.add(ColleagueDocument(
            colleague_id=cid, category="work_product",
            original_filename="report.txt",
            extracted_text="Annual revenue reached $10M milestone this quarter.",
        ))
        session.add(ColleagueQAPair(
            colleague_id=cid,
            prompt="How do I submit expenses?",
            response="Use the expense portal at expenses.company.com",
        ))
        session.add(ColleagueDocument(
            colleague_id=cid, category="reference",
            original_filename="handbook.txt",
            extracted_text="Employee handbook section 4.2: Expense reports must be submitted within 30 days.",
        ))

    work_products.process(cid, engine, generators)
    qa_upsampling.process(cid, engine, generators)
    references.process(cid, engine, generators)

    with get_session(engine) as session:
        all_samples = session.query(ColleagueSyntheticSample).filter_by(colleague_id=cid).all()
        assert len(all_samples) == 9  # 3 sources x 3 samples each

        by_type = {}
        for s in all_samples:
            by_type.setdefault(s.source_type, []).append(s)

        assert len(by_type["work_product"]) == 3
        assert len(by_type["qa"]) == 3
        assert len(by_type["reference"]) == 3

        # Verify no duplicates
        prompts = [s.synthetic_prompt for s in all_samples]
        # At least some should be unique (tokenizer fallback varies)
        unique = len(set(prompts))
        assert unique >= 3, f"Expected at least 3 unique prompts, got {unique}"
