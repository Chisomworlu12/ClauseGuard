from agents.aggregator import build_final_report
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

    final_report = build_final_report(
        db=db,
        run_id=run.id,
        judgments=judgments,
        contract_context=CONTRACT,
    )

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

    assert len(verifier_audits) == 1
    assert verifier_audits[0].clause_id is None

    print(f"\nrun_id={run.id}")
    print(f"verifier audit rows: {len(verifier_audits)}")
    print(f"verifier clause_id: {verifier_audits[0].clause_id}")

    db.close()


if __name__ == "__main__":
    main()