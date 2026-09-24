from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from config import settings
from retrieval.ensemble import get_hybrid_retriever


class RerankedRetriever(ContextualCompressionRetriever):
    """Hybrid retriever wrapped in a FlashRank cross-encoder reranker."""

    def __int__(self, base_compressor =None, base_retriever = None, **data):
        if base_compressor is None:
            base_compressor =FlashrankRerank(top_n= settings.retrieval_rerank_top_n)
        if base_retriever is None:
            base_retriever = get_hybrid_retriever()
        super().__init__(
            base_compressor = base_compressor,
            base_retriever = base_retriever,
            **data,
        )


def get_reranked_retriever() -> RerankedRetriever:

    """Return the fused retriever with FlashRank reranking applied."""

    reranker = FlashrankRerank(top_n= settings.retrieval_rerank_top_n) 

    return ContextualCompressionRetriever(
        base_compressor= reranker,
        base_retriever= get_hybrid_retriever(),
    )
