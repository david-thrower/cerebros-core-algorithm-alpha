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
