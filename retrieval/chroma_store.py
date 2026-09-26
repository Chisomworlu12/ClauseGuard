import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from retrieval.ingest import load_kb_documents, split_kb_documents


def get_embeddings()-> HuggingFaceEmbeddings:

    """Create the local Hugging Face embedding model used by Chroma."""

    return HuggingFaceEmbeddings(
        model_name = settings.retrieval_embedding_model,
        model_kwargs= {"device": settings.retrieval_embedding_device},
    )

def ingest_kb()-> Chroma:

    """Embed and upsert KB chunks into persistent Chroma, returning a LangChain Chroma store."""

    documents = split_kb_documents(load_kb_documents())
    ids= [document.id for document in documents]

    if any(not document_id for document_id in ids):
        raise ValueError("Every KB chunk must have an ID")

    if len(ids) != len(set(ids)):
        raise ValueError("KB chunk IDs must be unique")

    client = chromadb.PersistentClient(path=settings.retrieval_chroma_path)
    collection = client.get_or_create_collection(name=settings.retrieval_collection_name,)

    embeddings = get_embeddings()
    texts = [document.page_content for document in documents]

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=[document.metadata for document in documents],
        embeddings= embeddings.embed_documents(texts),
    )

    return Chroma(
        client=client,
        collection_name= settings.retrieval_collection_name,
        embedding_function= embeddings,
    )


def get_vectorstore()-> Chroma:
    
    """Open the persistent Chroma collection for dense vector retrieval."""

    return Chroma(
        collection_name= settings.retrieval_collection_name,
        persist_directory= settings.retrieval_chroma_path,
        embedding_function= get_embeddings(),
    )

def get_dense_retriever():

    """Return a LangChain retriever configured with the dense top-K setting."""

    return get_vectorstore().as_retriever(
        search_kwargs = { "k": settings.retrieval_dense_top_k}
    )

def retrieve_dense_with_scores(query: str):

    """Run dense Chroma retrieval and return documents with relevance scores."""
    
    return get_vectorstore().similarity_search_with_relevance_scores(
        query, k= settings.retrieval_log_top_n
    )



