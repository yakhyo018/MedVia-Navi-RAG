from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Every setting in one place, read from the environment (and `.env` locally).
    Prefix `NAVI_`: `NAVI_API_KEY`, `NAVI_QDRANT_URL`… — no clash with MedVia's own variables
    when both run on one server. A wrong value stops the app at boot, not at the first request.
    """

    model_config = SettingsConfigDict(
        env_prefix='NAVI_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    app_name: str = 'medvia-navi'
    environment: str = Field(default='development', pattern='^(development|test|production)$')
    log_level: str = 'INFO'

    # Service-to-service key: MedVia's API and batch call Navi with it. SecretStr keeps it out
    # of repr(), logs and tracebacks.
    api_key: SecretStr

    # `:memory:` = an in-process Qdrant for tests; `path:./qdrant_local` = embedded on disk
    # (no Docker, one process); otherwise the URL of a Qdrant server.
    qdrant_url: str = 'http://localhost:6333'
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = 'medvia_knowledge'
    qdrant_timeout_s: float = 5.0
    # multilingual-e5-small returns 384-dimensional vectors.
    embedding_dim: int = 384

    # `hash` / `fake`: no model, for tests and CI. `e5` / `local`: the real models.
    embedder: Literal['hash', 'e5'] = 'e5'
    embedding_model: str = 'intfloat/multilingual-e5-small'
    generator: Literal['fake', 'local'] = 'local'
    llm_model: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    # The LoRA adapter from R4 (`models/navi-lora`); empty = the base model.
    llm_adapter_path: str | None = None
    device: str = 'auto'
    max_new_tokens: int = 160

    top_k: int = Field(default=4, ge=1, le=10)
    # e5 cosine scores sit high (0.75–0.9 for related text): tuned in R5 on the eval set.
    min_score: float = Field(default=0.80, ge=0.0, le=1.0)

    @field_validator('api_key')
    @classmethod
    def api_key_is_long(cls, value: SecretStr) -> SecretStr:
        # 32+ characters: `openssl rand -hex 32` gives 64. A short key can be guessed.
        if len(value.get_secret_value()) < 32:
            raise ValueError('NAVI_API_KEY must be at least 32 characters')
        return value


@lru_cache
def get_settings() -> Settings:
    """One Settings per process; tests clear the cache to swap the environment."""
    return Settings()
