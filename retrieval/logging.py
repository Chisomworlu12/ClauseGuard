import uuid

from config import settings
from db.models import AgentAuditLog
from db.session import SessionLocal
from retrieval.bm25 import get_bm25_retriever
from retrieval.chroma_store import retrieve_dense_with_scores
from retrieval.ensemble import get_hybrid_retriever
from retrieval.reranker import get_reranked_retriever


def log_retrieval_stage(query: str) -> dict:
    
    """Run dense, BM25, and fused retrieval for one query and return per-stage IDs/ranks/scores."""

    top_n =settings.retrieval_log_top_n

    dense = retrieve_dense_with_scores(query)
    bm25 = get_bm25_retriever(top_k= top_n).retrieve_with_scores(query)
    fused = get_hybrid_retriever().invoke(query)[: int(settings.retrieval_fused_top_k)]
    reranked = get_reranked_retriever().invoke(query)[:settings.retrieval_rerank_top_n]

    dense_by_id = {doc.id: (rank, score) for rank, (doc, score) in enumerate(dense, 1)}
    bm25_by_id = {doc.id: (rank, score) for rank, (doc, score) in enumerate(bm25, 1)}
    fused_by_id = {doc.id: rank for rank, doc in enumerate(fused, 1)}

    return {
        "query" : query,
        "dense" : [{"id": d.id, "rank": r, "score": float(s)} for r, (d,s) in enumerate(dense, 1)],
        "bm25" : [{"id": d.id, "rank": r, "score": float(s)} for r, (d,s) in enumerate(bm25, 1)],
        "fused": [
            {
                "id": doc.id,
                "rank" : rank,
                "dense_rank": dense_by_id.get(doc.id, (None, None))[0],
                "bm25_rank": bm25_by_id.get(doc.id, (None, None))[0],
            }
            for rank, doc in enumerate(fused, 1)
        ],
        "reranked" : [
            {
                "id": doc.metadata.get("id") or doc.id,
                "rank": rank,
                "score": float(doc.metadata["relevance_score"]) if doc.metadata.get("relevance_score") is not None else None,
                "dense_rank" : dense_by_id.get(doc.metadata.get("id") or doc.id, (None, None))[0],
                "bm25_rank" : bm25_by_id.get(doc.metadata.get("id") or doc.id, (None, None))[0],
                "fused_rank" : fused_by_id.get(doc.metadata.get("id") or doc.id),

            }
            for rank, doc in enumerate(reranked, 1)
        ]
    }


def log_retrieval_audit(run_id: uuid.UUID, query: str) -> AgentAuditLog:

    """Persist the retrieval-stage scores for a run to agent_audit_log."""

    db= SessionLocal()
    try:
        audit = AgentAuditLog(
            run_id = run_id,
            agent = "retriever",
            input_summary = {"query": query},
            output_summary = log_retrieval_stage(query),
            model_used = settings.retrieval_embedding_model,
        )
        db.add(audit)
        db.commit()
        db.refresh(audit)
        return audit
    finally:
        db.close()