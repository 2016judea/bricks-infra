"""Settings for bricks-infra, loaded from environment / .env.

Mirrors the bricks repo convention (a single get_settings() accessor) so the two
codebases feel the same.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BRICKS_INFRA_", env_file=".env", extra="ignore"
    )

    # Local serving
    device: str = "mps"  # "mps" (Apple GPU), "cuda", or "cpu"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # LLM: local-only via Ollama. There is no cloud LLM tier by design — work that
    # exceeds the local model's quality stays on Opus in the bricks repo.
    ollama_url: str = "http://localhost:11434"
    local_llm: str = "qwen3:14b"

    # Image generation (cloud; Qwen-Image-2512 via fal.ai by default).
    # Qwen-Image is the open model that renders legible embedded text / brand
    # names best — the thing ad creative and mailers actually need.
    fal_api_key: str = ""
    image_base_url: str = "https://fal.run"
    image_model: str = "fal-ai/qwen-image"

    # Service
    host: str = "127.0.0.1"
    port: int = 8088


@lru_cache
def get_settings() -> Settings:
    return Settings()
