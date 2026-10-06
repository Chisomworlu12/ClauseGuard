import uuid

from sqlalchemy.orm import Session

from agents.aggregator import build_final_report, mark_escalated
from agents.judgment import judge_clause
from agents.segmenter import run_segmenter
from db.models import AnalysisRun, Clause


def run_analysis(
        contract_text: str,
        db: Session,
) -> tuple[AnalysisRun, str]:

    run = AnalysisRun(
        contract_text = contract_text,
        status= "running",
        judgment_model = "frontier",
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        clause_texts = run_segmenter(db, run.id, contract_text)
        judgments = []

        for position, text in enumerate(clause_texts, 1):
            clause = Clause(
                run_id = run.id,
                text = text,
                position= position,
                risk_level = "unable_to_assess",
                category = "pending",
                reason= "pending",
                confidence = "normal",
            )
            db.add(clause)
            db.commit()

            try:
                judgment = judge_clause(
                    db=db,
                    run_id= run.id,
                    clause_id= clause.id,
                    clause_text=text,
                    contract_context= contract_text
                )
            except Exception as error: # noqa: BLE001
                judgment = {
                    "risk_level": "unable_to_assess",
                    "category": "unable_to_assess",
                    "reason": f"unable_to_assess: {error}",
                    "confidence": "escalated",
                }

            judgments.append(
                {
                    **judgment,
                    "clause_id" : str(clause.id),
                    "position" : position,
                }
            )

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
            run_id= run.id,
            judgments= judgments,
            contract_context= contract_text,
        )

        run.overall_verdict = final_report
        run.status = "complete"
        db.commit()
        db.refresh(run)

        return run, final_report

    except Exception:
        run.status = "failed"
        db.commit()
        raise

           
            