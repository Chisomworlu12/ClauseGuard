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


settings = Settings()
