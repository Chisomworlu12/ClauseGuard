
from agents.judgment import judge_clause
from agents.segmenter import run_segmenter
from db.models import AgentAuditLog, AnalysisRun, Clause
from db.session import SessionLocal

CONTRACT ="""1. Term
This Agreement begins on the Effective Date and continues for twelve months

2. Termination
Either party may terminate this Agreement at any tie, with or without cause, upon no notice to the other party.

3. Payment
Client shall pay Provider within sixty days of receipt of invoice."""


def main():
    db = SessionLocal()

    run = AnalysisRun(contract_text= CONTRACT, status = "running", judgment_model = "frontier")
    db.add(run)
    db.commit()

    clause_text = run_segmenter(db, run.id, CONTRACT)
    print(f"segmented into {len(clause_text)} clauses")

    for position, text in enumerate(clause_text, 1):
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
        except Exception as error: # noqa: BLE001
            # A single clause failure must not stop the remaining clauses.
            judgment = {
                "risk_level": "unable_to_assess",
                "category": "unable_to_assess",
                "reason": f"unable_to_assess: {error}",
            }
            print(f"\n[clause {position}] failed, continuing...")
            print(f"  error: {error}")

        print(f"\n[clause {position}] {text[:60]}...")
        print(f"  risk_level: {judgment['risk_level']}")
        print(f"  category:   {judgment['category']}")
        print(f"  reason:     {judgment['reason']}")


    judgment_audits = (
        db.query(AgentAuditLog)
        .filter(
            AgentAuditLog.run_id == run.id,
            AgentAuditLog.agent == "judgment",
        )
        .all()
    )

    assert len(judgment_audits) == len(clause_text)
    assert all(row.clause_id is not None for row in judgment_audits)

    print(f"\nrun_id={run.id}")
    print(f"judgment audit rows: {len(judgment_audits)}")
    print(
        "audited clause ids:",
        [str(row.clause_id) for row in judgment_audits],
    )    


if __name__ == "__main__":
    main()