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
