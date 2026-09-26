from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str

    # DeepSeek is OpenAI-API-compatible, so the openai SDK works by pointing
    # base_url at DeepSeek instead of OpenAI.
    # Default "" (not None) so this is always a valid str even if unset -
    # check `if settings.deepseek_api_key` before using it.
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # FR-SEG-002: below this many clauses, or above this many words in a single
    # clause, the Segmenter falls back to an LLM call.
    segmenter_min_clauses: int = 2
    segmenter_max_clause_words: int = 800

    # Retrival
    retrieval_chunk_size: int = 500
    retrieval_chunk_overlap: int = 50
    retrieval_chunk_separators: list[str] = ["\n\n", "\n", " ", ""]
    retrieval_embedding_model: str= "sentence-transformers/all-MiniLM-L6-V2"
    retrieval_embedding_device: str= "cpu"
    retrieval_chroma_path: str= ".chroma"
    retrieval_collection_name: str= "clauseguard_kb"
    retrieval_dense_top_k: int= 5
    retrieval_bm25_top_k: int= 5
    retrieval_dense_weight: float= 0.5
    retrieval_bm25_weight: float= 0.5
    retrieval_fused_top_k: int= 10
    retrieval_log_top_n: int =10
    retrieval_rerank_model: str ="ms-marco-MultiBERT-L-12"
    retrieval_rerank_top_n: int = 5
  




settings = Settings()
