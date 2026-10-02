"""Evaluate Judgment Agent accuracy against the fixed EvalCase labels."""

from agents.judgment import judge_clause
from db.models import AnalysisRun, Clause
from db.session import SessionLocal
from eval.cases import EVAL_CASES
from eval.persistence import persist_eval_run
from retrieval.reranker import get_reranked_retriever


def score_judgment(case, judgment:dict) -> dict:

    """Compare one model judgment with the expected EvalCase labels."""

    risk_correct = (
        judgment.get("risk_level") == case.expected_risk_level
    )
    category_correct = ( 
        judgment.get("category") == case.expected_category
    )

    return {
        "risk_correct" : risk_correct,
        "category_correct": category_correct,
        "exact_correct": risk_correct and category_correct,
    }


def main():

    """Run the Judgment Agent evaluation and persist its result."""

    db = SessionLocal()
    retriever = get_reranked_retriever()

    run = AnalysisRun(
        contract_text = "phase 7 judgment evaluation run",
        status = "running",
        judgment_model = "frontier"
    )
    db.add(run)
    db.commit()

    risk_correct = 0
    category_correct = 0
    exact_correct = 0

    for position, case in enumerate(EVAL_CASES, 1):
        clause = Clause(
            run_id = run.id,
            text = case.clause_text,
            position = position,
            risk_level = "unable_to_assess",
            category = "pending",
            reason = "pending",
            confidence = "normal",
        )
        db.add(clause)
        db.commit()

        retrieved_docs = retriever.invoke(case.clause_text)

        judgment = judge_clause(
            db = db,
            run_id = run.id,
            clause_id= clause.id,
            clause_text= case.clause_text,
            retrieved_docs= retrieved_docs,
            contract_context= "Evaluation contract clause" ,
        )

        scores = score_judgment(case, judgment)

        risk_correct += scores["risk_correct"]
        category_correct += scores["category_correct"]
        exact_correct += scores["exact_correct"]

        print(
            f"{case.case_id}: "
            f"risk={judgment.get('risk_level')} "
            f"category={judgment.get('category')} "
            f"exact={scores['exact_correct']}"
        )

    case_count = len(EVAL_CASES)

    risk_accuracy = risk_correct / case_count
    category_accuracy = category_correct / case_count
    exact_accuracy = exact_correct / case_count

    persist_eval_run(
        db,
        model = "frontier",
        retrieval_enabled = True,
        accuracy = exact_accuracy,
        mrr = 0.0,
        ndcg = 0.0,
    )

    run.status = "complete"
    db.commit()

    print("\nJudgment evaluation results")
    print(f"cases evaluated: {case_count}")
    print(f"risk accuracy: {risk_correct / case_count:.4f}")
    print(f"category accuracy: {category_correct / case_count:.4f}")
    print(f"exact accuracy: {exact_correct / case_count:.4f}")
    print(f"risk accuracy: {risk_accuracy:.4f}")
    print(f"category accuracy: {category_accuracy:.4f}")
    print(f"exact accuracy: {exact_accuracy:.4f}")

    run.status = "complete"
    db.commit()
    db.close()

if __name__ == "__main__":
    main()