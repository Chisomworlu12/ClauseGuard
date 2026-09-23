import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

DEFAULT_KB_PATH = Path(__file__).resolve().parents[1] / "data" / "kb_entries.json"


def load_kb_documents(kb_path: Path = DEFAULT_KB_PATH) -> list[Document]:
    """Load KB entries as LangChain Documents with stable IDs and categories."""
    entries = json.loads(kb_path.read_text(encoding="utf-8"))

    if not isinstance(entries, list):
        raise TypeError("Knowledge base must contain a JSON list")

    documents: list[Document] = []

    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Each knowledge-base entry must be an object")

        required = {"id", "category", "reference_text"}
        missing = required - entry.keys()

        if missing:
            raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")

        documents.append(
            Document(
                page_content=str(entry["reference_text"]),
                id=str(entry["id"]),
                metadata={
                    "id": str(entry["id"]),
                    "category": str(entry["category"]),
                },
            )
        )

    return documents


def split_kb_documents(documents: list[Document]) -> list[Document]:

    """Split KB documents into configured chunks while preserving source IDs and categories."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.retrieval_chunk_size,
        chunk_overlap=settings.retrieval_chunk_overlap,
        separators=settings.retrieval_chunk_separators,
    )

    chunks: list[Document] = []

    for document in documents:
        source_id = document.id or document.metadata.get("id")
        if not source_id:
            raise ValueError("Every knowledge-base document must have an ID")

        split_documents = splitter.split_documents([document])

        for index, chunk in enumerate(split_documents):
            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    id=f"{source_id}--chunk-{index + 1}",
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                    },
                )
            )

    return chunks
