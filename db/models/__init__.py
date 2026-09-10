"""Database models for the six tables described in SRS section 2.

Everything inherits from a single Base so that Alembic can pick up every table
from Base.metadata when it autogenerates migrations.
"""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models. migrations/env.py points target_metadata at this."""


class AnalysisRun(Base):
    """A single run of the pipeline over one contract.

    This is the parent record. Clauses and audit-log rows both reference it.
    """

    __tablename__ = "analysis_runs"

    # Generate the id in Python instead of letting the DB assign a serial int.
    # The agents need to reference the run while the pipeline is still going, so
    # the id has to exist before the row is committed.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # The contract the user submitted. We only keep it for this run's history
    # (see the privacy note in SRS section 13).
    contract_text: Mapped[str] = mapped_column(Text)

    # The Verifier/Aggregator fills this in at the end. It stays null if the run
    # fails before it gets there.
    overall_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Using a plain string with a CHECK constraint instead of a Postgres ENUM.
    # If we ever need another status value, that's a normal migration rather than
    # an ALTER TYPE, which is awkward to do safely.
    status: Mapped[str] = mapped_column(String(20))
    judgment_model: Mapped[str] = mapped_column(String(20))

    # Let Postgres set the timestamp so it's the same regardless of which process
    # inserts the row.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('running','complete','failed')", name="ck_runs_status"
        ),
        CheckConstraint(
            "judgment_model IN ('frontier','fine_tuned')", name="ck_runs_model"
        ),
    )


class Clause(Base):
    """One clause pulled out of a contract, together with the risk judgment for it."""

    __tablename__ = "clauses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # No cascade on delete. We don't delete runs in this project, so the default
    # behaviour (block the delete if clauses still point at it) is what we want.
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_runs.id"))

    text: Mapped[str] = mapped_column(Text)

    # The SRS calls this "order". Renamed to "position" because ORDER is a
    # reserved word in SQL and quoting it everywhere is a pain.
    position: Mapped[int] = mapped_column(Integer)

    risk_level: Mapped[str] = mapped_column(String(20))
    category: Mapped[str] = mapped_column(String)
    reason: Mapped[str] = mapped_column(Text)

    # Normally "normal". The Verifier changes it to "escalated" when a judgment
    # looks low-confidence, so the frontend can flag it for a human to check
    # (FR-AGG-004).
    confidence: Mapped[str] = mapped_column(String(20))

    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('low','medium','high','unable_to_assess')",
            name="ck_clauses_risk",
        ),
        CheckConstraint(
            "confidence IN ('normal','escalated')", name="ck_clauses_confidence"
        ),
    )


class AgentAuditLog(Base):
    """The audit trail (SRS section 2.3).

    One row for every decision an agent makes, not one row per run. This is what
    lets us go back and see why the pipeline produced a given result.
    """

    __tablename__ = "agent_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_runs.id"))

    # Null for agents that work at the run level (Segmenter, Verifier/Aggregator).
    # Set for agents that work on a single clause (Retriever, Judgment).
    clause_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("clauses.id"), nullable=True
    )

    agent: Mapped[str] = mapped_column(String(20))

    # Stored as JSON because the useful contents differ by agent. The Retriever,
    # for example, records which reference entries it matched and their scores.
    input_summary: Mapped[dict] = mapped_column(JSONB)
    output_summary: Mapped[dict] = mapped_column(JSONB)

    # Null when the step didn't call a model, e.g. the rule-based segmentation path.
    model_used: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "agent IN ('segmenter','retriever','judgment','verifier')",
            name="ck_audit_agent",
        ),
    )


class ReferenceKbEntry(Base):
    """A hand-written note on what's normal or a red flag for a clause category.

    Postgres stores the text. The embedding for it lives in ChromaDB, not here.
    """

    __tablename__ = "reference_kb_entries"

    # A short readable id we assign by hand, like "termination-01". Easier to work
    # with than a UUID while writing and reviewing the ~40-60 entries.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String)
    reference_text: Mapped[str] = mapped_column(Text)


class EvalRun(Base):
    """One result from the eval harness.

    Saved over time so we can track accuracy, MRR and nDCG as a trend instead of
    a single number in the README (SRS section 2.5).
    """

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    model: Mapped[str] = mapped_column(String(20))

    # model and retrieval_enabled together cover the 2x2 comparison from PRD
    # section 10.2: frontier vs fine-tuned, retrieval on vs off.
    retrieval_enabled: Mapped[bool] = mapped_column(Boolean)

    # These are scores between 0 and 1, so Numeric rather than Integer.
    accuracy: Mapped[float] = mapped_column(Numeric)
    mrr: Mapped[float] = mapped_column(Numeric)
    ndcg: Mapped[float] = mapped_column(Numeric)

    # The commit the eval ran against, so any number here is reproducible.
    commit_hash: Mapped[str] = mapped_column(String)

    __table_args__ = (
        CheckConstraint("model IN ('frontier','fine_tuned')", name="ck_eval_model"),
    )


class RateLimitLog(Base):
    """Counts requests per IP per day for the rate limit (SRS sections 2.6 and 7.1).

    There are no user accounts in this project, so this simple counter takes the
    place of proper auth for abuse protection.
    """

    __tablename__ = "rate_limit_log"

    # Composite primary key: one row per IP per calendar day. No separate id column.
    ip_address: Mapped[str] = mapped_column(String, primary_key=True)

    # Date, not a timestamp. The limit resets at midnight, not on a rolling 24-hour
    # window (FR-RATE-002).
    request_date: Mapped[date] = mapped_column(Date, primary_key=True)

    request_count: Mapped[int] = mapped_column(Integer)
