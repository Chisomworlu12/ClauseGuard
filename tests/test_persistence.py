from datetime import UTC, datetime
from uuid import uuid4

from db.models import (
    AgentAuditLog,
    AnalysisRun,
    Clause,
    EvalRun,
    RateLimitLog,
    ReferenceKbEntry,
)
from db.session import SessionLocal


def test_all_srs_tables_accept_valid_writes():
    """Verify valid ORM rows can be written to every SRS database table."""

    db = SessionLocal()

    run = AnalysisRun(
        contract_text="Persistence test contract",
        status="running",
        judgment_model="frontier",
    )

    reference = ReferenceKbEntry(
        id=f"test-persistence-{uuid4()}",
        category="termination",
        reference_text="Test termination reference.",
    )

    evaluation = EvalRun(
        model="frontier",
        retrieval_enabled=True,
        accuracy=0.8,
        mrr=0.7,
        ndcg=0.75,
        commit_hash="test-commit",
    )

    rate_limit = RateLimitLog(
        ip_address=f"test-{uuid4()}",
        request_date=datetime.now(UTC).date(),
        request_count=1,
    )

    committed = False

    try:
        db.add(run)
        db.flush()

        clause = Clause(
            run_id=run.id,
            text="Either party may terminate with thirty days notice.",
            position=1,
            risk_level="medium",
            category="termination",
            reason="Persistence test judgment.",
            confidence="normal",
        )
        db.add(clause)
        db.flush()

        audit = AgentAuditLog(
            run_id=run.id,
            clause_id=clause.id,
            agent="judgment",
            input_summary={"clause_text": clause.text},
            output_summary={
                "risk_level": clause.risk_level,
                "category": clause.category,
                "reason": clause.reason,
            },
            model_used="frontier",
        )

        db.add_all([audit, reference, evaluation, rate_limit])
        db.commit()
        committed = True

        assert db.get(AnalysisRun, run.id) is not None
        assert db.get(Clause, clause.id) is not None
        assert db.get(AgentAuditLog, audit.id) is not None
        assert db.get(ReferenceKbEntry, reference.id) is not None
        assert db.get(EvalRun, evaluation.id) is not None
        assert db.get(
            RateLimitLog,
            {
                "ip_address": rate_limit.ip_address,
                "request_date": rate_limit.request_date,
            },
        ) is not None

    finally:
        if committed:
            db.delete(audit)
            db.flush()

            db.delete(clause)
            db.flush()

            db.delete(run)
            db.delete(reference)
            db.delete(evaluation)
            db.delete(rate_limit)
            db.commit()
        else:
            db.rollback()

        db.close()