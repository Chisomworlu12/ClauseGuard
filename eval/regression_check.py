"""Verify that the evaluation harness detects a deliberately bad retriever."""

from eval.cases import EVAL_CASES
from eval.retrieval_eval import evaluate_retriever
from retrieval.chroma_store import get_dense_retriever


class EmptyRetriever:
    """Deliberately broken retriever that returns no results."""

    def invoke(self, query: str) -> list:
        return []

def main():
    """Compare the normal retriever with a deliberately failed retriever."""

    cases = [
        case for case in EVAL_CASES
        if case.expected_reference_ids
    ]

    baseline_results = evaluate_retriever(
        get_dense_retriever(),
        cases,
    )

    degraded_results =evaluate_retriever(
        EmptyRetriever(),
        cases
    )

    print("Baseline retrieval")
    print(f"MRR: {baseline_results['mrr']:.4f}")
    print(f"nDCG: {baseline_results['ndcg']:.4f}")

    print("\nDeliberately degraded retrieval")
    print(f"MRR: {degraded_results['mrr']:.4f}")
    print(f"nDCG: {degraded_results['ndcg']:.4f}")

    assert degraded_results["mrr"] < baseline_results["mrr"]
    assert degraded_results["ndcg"] < baseline_results["ndcg"]

    print("\nRegression detected successfully.")


if __name__ == "__main__":
    main()