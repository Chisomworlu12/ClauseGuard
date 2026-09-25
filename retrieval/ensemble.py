from langchain_classic.retrievers import EnsembleRetriever

from config import settings
from retrieval.bm25 import get_bm25_retriever
from retrieval.chroma_store import get_dense_retriever


def get_hybrid_retriever()-> EnsembleRetriever:

    """Combine dense and BM25 retrievers using configurable RRF fusion weights."""
    
    return EnsembleRetriever(
        retrievers=[get_dense_retriever(), get_bm25_retriever()],
        weights=[
            settings.retrieval_dense_weight,
            settings.retrieval_bm25_weight,
        ],
    )