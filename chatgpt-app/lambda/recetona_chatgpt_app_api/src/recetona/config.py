from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="RECETONA_",
        extra="ignore",
    )

    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    rag_cache_dir: Path = PROJECT_ROOT / "rag_cache"
    images_dir: Path = PROJECT_ROOT / "images"
    catalog_csv_path: Path = PROJECT_ROOT / "data" / "catalog.csv"
    chunks_csv_path: Path = PROJECT_ROOT / "rag_cache" / "chunks.csv"
    embeddings_path: Path = PROJECT_ROOT / "rag_cache" / "embeddings.npy"
    embeddings_hash_path: Path = (
        PROJECT_ROOT / "rag_cache" / "embeddings.sha256"
    )
    scrape_checkpoint_path: Path = (
        PROJECT_ROOT / "data" / "scrape_checkpoint.json"
    )
    legacy_excel_path: Path = PROJECT_ROOT / "mercadona_data.xlsx"
    notebook_path: Path = PROJECT_ROOT / "mercadona_rag_notebook.ipynb"

    openai_api_key: str | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-5.4-nano"
    reasoning_effort: str = "none"

    http_host: str = "127.0.0.1"
    http_port: int = 8787
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8788
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "null",
            "http://127.0.0.1:8787",
            "http://localhost:8787",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    )
    debug: bool = False

    mercadona_timeout_seconds: int = 20
    mercadona_max_retries: int = 5
    openai_max_retries: int = 4
    openai_retry_base_seconds: float = 1.0
    image_download_workers: int = 20
    max_images_per_row: int = 2
    log_level: str = "INFO"

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "root_dir",
        "data_dir",
        "rag_cache_dir",
        "images_dir",
        "catalog_csv_path",
        "chunks_csv_path",
        "embeddings_path",
        "embeddings_hash_path",
        "scrape_checkpoint_path",
        "legacy_excel_path",
        "notebook_path",
        mode="before",
    )
    @classmethod
    def _ensure_path(cls, value: object) -> object:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        return value

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rag_cache_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str | None = None) -> None:
    resolved_level = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s event=%(message)s",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
