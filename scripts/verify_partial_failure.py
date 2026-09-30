from agents.judgment import judge_clause
from agents.segmenter import run_segmenter
from db.models import AgentAuditLog, AnalysisRun, Clause
from db.session import SessionLocal

CONTRACT = """1. Term
This Agreement begins on the Effective Date and continues for twelve months.

2. Termination
Either party may terminate this Agreement at any time, with or without cause,
upon no notice to the other party.

3. Payment
Client shall pay Provider within sixty days of receipt of invoice."""


class ForcedMidRunFailure(RuntimeError):
    """Failure deliberately injected to verify partial persistence."""


def main():
    db = SessionLocal()

    run = AnalysisRun(
        contract_text=CONTRACT,
        status="running",
        judgment_model="frontier",
    )
    db.add(run)
    db.commit()

    try:
        clause_texts = run_segmenter(db, run.id, CONTRACT)

        for position, text in enumerate(clause_texts, 1):
            clause = Clause(
                run_id=run.id,
                text=text,
                position=position,
                risk_level="unable_to_assess",
                category="pending",
                reason="pending",
                confidence="normal",
            )
            db.add(clause)
            db.commit()

            judgment = judge_clause(
                db=db,
                run_id=run.id,
                clause_id=clause.id,
                clause_text=text,
                contract_context=CONTRACT,
            )

            clause.risk_level = judgment["risk_level"]
            clause.category = judgment["category"]
            clause.reason = judgment["reason"]
            clause.confidence = (
                "escalated"
                if judgment["risk_level"] == "unable_to_assess"
                else "normal"
            )
            db.commit()

            if position == 1:
                raise ForcedMidRunFailure(
                    "deliberate failure after first clause"
                )

    except ForcedMidRunFailure as error:
        db.rollback()

        run.status = "failed"
        db.commit()

        print(f"forced failure: {error}")

    saved_run = db.get(AnalysisRun, run.id)

    saved_clauses = (
        db.query(Clause)
        .filter(Clause.run_id == run.id)
        .order_by(Clause.position)
        .all()
    )

    saved_audits = (
        db.query(AgentAuditLog)
        .filter(AgentAuditLog.run_id == run.id)
        .all()
    )

    audit_agents = [audit.agent for audit in saved_audits]

    assert saved_run.status == "failed"
    assert len(saved_clauses) == 1
    assert "segmenter" in audit_agents
    assert "judgment" in audit_agents

    print(f"run status: {saved_run.status}")
    print(f"saved clauses: {len(saved_clauses)}")
    print(f"saved audit rows: {len(saved_audits)}")
    print(f"audit agents: {audit_agents}")

    db.close()


if __name__ == "__main__":
    main()