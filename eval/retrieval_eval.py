"""Evaluate dense-only and hybrid-plus-reranker retrieval quality.

This module computes MRR and nDCG against the fixed EvalCase reference labels.
Ambiguous cases without known reference IDs are excluded from retrieval metrics.
"""

from math import log2

from db.session import SessionLocal
from eval.cases import EVAL_CASES
from eval.persistence import persist_eval_run
from retrieval.chroma_store import get_dense_retriever
from retrieval.reranker import get_reranked_retriever


def document_source_id(document) -> str:
    """Return the original KB entry ID for a retrieved document."""

    return document.metadata.get("id") or document.id

def reciprocal_rank(retrieved_ids: list[str], expected_ids: tuple[str, ...]) -> float:
    """Return the reciprocal rank of the first relevant retrieved entry."""

    expected = set(expected_ids)

    for position, document_id in enumerate(retrieved_ids, 1):
        if document_id in expected:
            return 1/ position

    return 0.0

def ndcg(retrieved_ids: list[str], expected_ids: tuple[str, ...]) -> float:
    """Calculate normalized discounted cumulative gain for one query."""
    expected = set(expected_ids)

    gains = [
        1 if document_id in expected else 0
        for document_id in retrieved_ids
    ]

    dcg = sum(
        gain / log2(position +1)
        for position, gain in enumerate(gains, 1)
    )

    ideal_gains = [1] * min(len(expected), len(retrieved_ids))
    ideal_dcg = sum(
        gain / log2(position + 1)
        for position, gain in enumerate(ideal_gains, 1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0

def evaluate_retriever(retriever, cases) -> dict:
    """Run one retriever across evaluation cases and return MRR and nDCG."""

    reciprocal_ranks = []
    ndcg_scores = []
    hits = 0

    for case in cases:
        documents = retriever.invoke(case.clause_text)

        retrieved_ids = [
            document_source_id(document)
            for document in documents
        ]

        rank = reciprocal_rank(
            retrieved_ids,
            case.expected_reference_ids,
        )

        reciprocal_ranks.append(rank)

        if rank > 0:
            hits += 1
        ndcg_scores.append(
            ndcg(
                retrieved_ids,
                case.expected_reference_ids
            )
        )

    return {
        "case_count" : len(cases),
        "accuracy": hits / len(cases),
        "mrr" : sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg": sum(ndcg_scores) / len(ndcg_scores)
    }


    # Retrieval metrics require a known correct reference match.
    # Ambiguous cases are evaluated later by the judgment harness.
def main():
    """Compare dense-only retrieval with hybrid retrieval and reranking."""

    retrieval_cases = [
        case for case in EVAL_CASES if case.expected_reference_ids
    ]

    dense_retriever = get_dense_retriever()
    hybrid_reranked_retriever = get_reranked_retriever()

    dense_results = evaluate_retriever(
        dense_retriever,
        retrieval_cases
    )
    hybrid_results = evaluate_retriever(
        hybrid_reranked_retriever,
        retrieval_cases,
    )

    db = SessionLocal()

    try:
        persist_eval_run(
            db,
            model = "frontier",
            retrieval_enabled= False,
            accuracy= dense_results["accuracy"],
            mrr = dense_results["mrr"],
            ndcg= dense_results["ndcg"]
        )

        persist_eval_run(
            db,
            model = "frontier",
            retrieval_enabled= True,
            accuracy= hybrid_results["accuracy"],
            mrr = hybrid_results["mrr"],
            ndcg = hybrid_results["ndcg"]
        )
    finally:
        db.close()

    print(f"retrieval cases evaluated: {len(retrieval_cases)}")

    print("\nDense-only baseline")
    print(f"Hit rate: {dense_results['accuracy']:.4f}")
    print(f"MRR: {dense_results['mrr']:.4f}")
    print(f"nDCG: {dense_results['ndcg']:.4f}")

    print("\nHybrid + reranker")
    print(f"Hit rate: {hybrid_results['accuracy']:.4f}")
    print(f"MRR: {hybrid_results['mrr']:.4f}")
    print(f"nDCG: {hybrid_results['ndcg']:.4f}")


if __name__ == "__main__":
    main()