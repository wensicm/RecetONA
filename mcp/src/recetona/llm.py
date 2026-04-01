from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
from openai import OpenAI

from .config import Settings


LOGGER = logging.getLogger(__name__)


class OpenAIBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def require_client(self) -> OpenAI:
        if self.client is None:
            raise RuntimeError("Falta OPENAI_API_KEY. Configura OPENAI_API_KEY o RECETONA_OPENAI_API_KEY.")
        return self.client

    def _retry(self, operation_name: str, callback):
        last_error: Exception | None = None
        for attempt in range(1, self.settings.openai_max_retries + 1):
            try:
                return callback()
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.openai_max_retries:
                    break
                wait_seconds = self.settings.openai_retry_base_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "openai_retry operation=%s attempt=%s wait_seconds=%.2f error=%s",
                    operation_name,
                    attempt,
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)
        raise RuntimeError(f"OpenAI {operation_name} failed: {last_error}") from last_error

    def embed_texts(self, texts: Sequence[str], model: str | None = None, batch_size: int = 64) -> np.ndarray:
        client = self.require_client()
        embed_model = model or self.settings.embed_model
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = self._retry(
                "embeddings.create",
                lambda batch=batch: client.embeddings.create(model=embed_model, input=batch),
            )
            vectors.extend(item.embedding for item in response.data)

        array = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
        return array / norms

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        client = self.require_client()
        chat_model = model or self.settings.chat_model
        response = self._retry(
            "responses.create",
            lambda: client.responses.create(model=chat_model, input=prompt),
        )
        return (response.output_text or "").strip()

    def generate_response(self, prompt: str, model: str | None = None) -> Any:
        client = self.require_client()
        chat_model = model or self.settings.chat_model
        return self._retry(
            "responses.create",
            lambda: client.responses.create(model=chat_model, input=prompt),
        )
