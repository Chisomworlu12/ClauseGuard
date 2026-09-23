import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from config import settings
from retrieval.ingest import load_kb_documents, split_kb_documents


def tokenize(text:str) -> list[str]:

    """Normalize text into lowercase word tokens for BM25."""

    return re.findall(r"\w+", text.lower())

class CustomBM25Retriever(BaseRetriever):
    documents: list[Document]
    top_k: int
    vectorizer: Any = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,) -> list[Document]:

        """Rank KB documents by BM25 relevance and return the configured top-K."""

        scores = self.vectorizer.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key = lambda index: scores[index],
            reverse =  True
        )

        return [
            self.documents[index]
            for index in ranked_indices[: self.top_k]
            if scores[index] > 0
        ]


def get_bm25_retriever()-> CustomBM25Retriever:

    """Build a BM25 retriever from the same split KB documents used by dense retrieval."""

    documents = split_kb_documents(load_kb_documents())

    if not documents:
        raise ValueError("Cannot build BM25 retiever without KB documents")

    return CustomBM25Retriever(
        documents=documents,
        top_k= settings.retrieval_bm25_top_k,
        vectorizer=BM25Okapi(
            [tokenize(document.page_content) for document in documents]
        ),
    )