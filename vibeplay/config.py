"""Central configuration for vibeplay."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


LIBRARY_COLUMNS: Tuple[str, ...] = (
    "Added At",
    "Track Name",
    "Artists",
    "Album",
    "Id",
)

VIBES_COLUMNS: Tuple[str, ...] = (
    "Id",
    "VibeSentence",
    "model",
    "created_at",
    "prompt_tokens",
    "completion_tokens",
)

PLAYLIST_EXTRA_COLUMNS: Tuple[str, ...] = ("VibeSentence", "Similarity")


class Settings(BaseSettings):
    """Runtime configuration loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
    )

    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(
        default="models/gemini-2.5-flash-lite",
        description="Gemini model identifier for vibe generation.",
    )
    gemini_api_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Base URL for Gemini API.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model for embeddings.",
    )
    cache_dir: Path = Field(
        default=Path(".cache") / "vibes",
        description="Directory for per-track vibe cache.",
    )
    batch_size: int = Field(default=25, description="Batch size for LLM and encoder.")
    rpm: int = Field(default=15, description="Requests allowed per minute for Gemini API.")
    tpm: int = Field(default=250_000, description="Tokens allowed per minute for Gemini API.")
    rpd: int = Field(default=1_000, description="Requests allowed per day for Gemini API.")
    llm_timeout_seconds: int = Field(default=30, description="HTTP timeout for Gemini API calls.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()

