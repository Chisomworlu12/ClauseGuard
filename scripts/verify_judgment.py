import uuid

from agents.aggregator import build_final_report, mark_escalated
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


def main():
    db = SessionLocal()

    run = AnalysisRun(
        contract_text=CONTRACT,
        status="running",
        judgment_model="frontier",
    )
    db.add(run)
    db.commit()

    clause_texts = run_segmenter(db, run.id, CONTRACT)
    print(f"segmented into {len(clause_texts)} clauses")

    judgments = []

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

        try:
            judgment = judge_clause(
                db=db,
                run_id=run.id,
                clause_id=clause.id,
                clause_text=text,
                contract_context=CONTRACT,
            )
        except Exception as error:  # noqa: BLE001
            judgment = {
                "risk_level": "unable_to_assess",
                "category": "unable_to_assess",
                "reason": f"unable_to_assess: {error}",
            }
            print(f"\n[clause {position}] failed, continuing...")
            print(f"  error: {error}")

        judgments.append(
            {
                **judgment,
                "clause_id": str(clause.id),
                "position": position,
            }
        )

        print(f"\n[clause {position}] {text[:60]}...")
        print(f"  risk_level: {judgment['risk_level']}")
        print(f"  category:   {judgment['category']}")
        print(f"  reason:     {judgment['reason']}")
    judgments = mark_escalated(judgments)
    
    for judgment in judgments:
        clause = db.get(
            Clause,
            uuid.UUID(judgment["clause_id"]),
        )

        if clause is None:
            raise RuntimeError(
                f"Clause not found: {judgment['clause_id']}"
            )

        clause.risk_level = judgment["risk_level"]
        clause.category = judgment["category"]
        clause.reason = judgment["reason"]
        clause.confidence = judgment["confidence"]

    db.commit()


    final_report = build_final_report(
        db=db,
        run_id=run.id,
        judgments=judgments,
        contract_context=CONTRACT,
    )

    run.overall_verdict = final_report
    run.status = "complete"
    db.commit()

    print("\nFINAL REPORT")
    print(final_report)

    verifier_audits = (
        db.query(AgentAuditLog)
        .filter(
            AgentAuditLog.run_id == run.id,
            AgentAuditLog.agent == "verifier",
        )
        .all()
    )

    all_audits = (
        db.query(AgentAuditLog)
        .filter(AgentAuditLog.run_id == run.id)
        .all()
    )

    audit_agents = [audit.agent for audit in all_audits]

    assert audit_agents.count("segmenter") == 1
    assert audit_agents.count("judgment") == len(clause_texts)
    assert audit_agents.count("verifier") == 1

    print(f"total audit rows: {len(all_audits)}")
    print(f"audit agents: {audit_agents}")

    assert len(verifier_audits) == 1
    assert verifier_audits[0].clause_id is None

    saved_run = db.get(AnalysisRun, run.id)

    saved_clauses = (
        db.query(Clause)
        .filter(Clause.run_id == run.id)
        .order_by(Clause.position)
        .all()
    )

    assert saved_run.status == "complete"
    assert saved_run.overall_verdict
    assert len(saved_clauses) == len(clause_texts)
    assert all(clause.category != "pending" for clause in saved_clauses)
    assert all(clause.reason != "pending" for clause in saved_clauses)

    print(f"analysis status: {saved_run.status}")
    print(f"saved clauses: {len(saved_clauses)}")
    print(f"overall verdict saved: {bool(saved_run.overall_verdict)}")

    db.close()


if __name__ == "__main__":
    main()