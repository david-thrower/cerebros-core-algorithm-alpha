from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Colleague(Base):
    __tablename__ = "colleagues"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    status = Column(String(50), default="draft")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    documents = relationship("ColleagueDocument", back_populates="colleague", cascade="all, delete-orphan")
    qa_pairs = relationship("ColleagueQAPair", back_populates="colleague", cascade="all, delete-orphan")
    synthetic_samples = relationship("ColleagueSyntheticSample", back_populates="colleague", cascade="all, delete-orphan")


class ColleagueDocument(Base):
    __tablename__ = "colleague_documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    colleague_id = Column(Integer, ForeignKey("colleagues.id"), nullable=False)
    category = Column(String(50), nullable=False)
    original_filename = Column(String(512), nullable=False)
    extracted_text = Column(Text, default="")
    processing_status = Column(String(50), default="pending")
    platform = Column(String(50), default="")
    user_identity = Column(String(255), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    colleague = relationship("Colleague", back_populates="documents")


class ColleagueQAPair(Base):
    __tablename__ = "colleague_qa_pairs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    colleague_id = Column(Integer, ForeignKey("colleagues.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    colleague = relationship("Colleague", back_populates="qa_pairs")


class ColleagueSyntheticSample(Base):
    __tablename__ = "colleague_synthetic_samples"
    id = Column(Integer, primary_key=True, autoincrement=True)
    colleague_id = Column(Integer, ForeignKey("colleagues.id"), nullable=False)
    source_type = Column(String(50), nullable=False)
    source_id = Column(Integer, nullable=True)
    synthetic_prompt = Column(Text, nullable=False)
    synthetic_reasoning = Column(Text, default="")
    synthetic_response = Column(Text, default="")
    prompt_style = Column(String(100), default="")
    approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    colleague = relationship("Colleague", back_populates="synthetic_samples")
