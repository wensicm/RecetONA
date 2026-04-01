from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from openai import OpenAI

from .catalog import LEGACY_EXCEL_MAPPING, ensure_catalog_schema
from .config import Settings
from .index import build_chunks_df, compute_chunks_hash, save_chunks_df

LOGGER = logging.getLogger(__name__)

EMBEDDINGS_METADATA_FILENAME = "embeddings.meta.json"
EMBEDDINGS_PARTIAL_DIRNAME = "embeddings.partial"

_THREAD_LOCAL = threading.local()


def load_legacy_excel_catalog(settings: Settings) -> pd.DataFrame:
    if not settings.legacy_excel_path.exists():
        raise FileNotFoundError(
            f"No existe el Excel legacy en {settings.legacy_excel_path}."
        )

    legacy_dataframe = pd.read_excel(settings.legacy_excel_path)
    migrated_data: dict[str, Any] = {}
    for source_column, target_column in LEGACY_EXCEL_MAPPING.items():
        if source_column in legacy_dataframe.columns:
            migrated_data[target_column] = legacy_dataframe[source_column]

    migrated_dataframe = pd.DataFrame(migrated_data)
    migrated_dataframe["source_updated_at"] = ""
    return ensure_catalog_schema(migrated_dataframe)


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=np.float32)
    if normalized.size == 0:
        return normalized
    norms = np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-12
    return normalized / norms


def _metadata_path(settings: Settings) -> Path:
    return settings.rag_cache_dir / EMBEDDINGS_METADATA_FILENAME


def _partial_dir(settings: Settings) -> Path:
    return settings.rag_cache_dir / EMBEDDINGS_PARTIAL_DIRNAME


def _partial_manifest_path(settings: Settings) -> Path:
    return _partial_dir(settings) / "manifest.json"


def _batch_file_path(settings: Settings, batch_index: int) -> Path:
    return _partial_dir(settings) / f"batch_{batch_index:05d}.npy"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_manifest(
    *,
    chunks_hash: str,
    model: str,
    dimensions: int | None,
    row_count: int,
    batch_size: int,
    total_batches: int,
) -> dict[str, Any]:
    return {
        "chunks_hash": chunks_hash,
        "model": model,
        "dimensions": dimensions,
        "row_count": row_count,
        "batch_size": batch_size,
        "total_batches": total_batches,
    }


def _build_metadata(
    *,
    chunks_hash: str,
    model: str,
    dimensions: int | None,
    row_count: int,
    output_dimensions: int,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    return {
        "chunks_hash": chunks_hash,
        "model": model,
        "dimensions": dimensions,
        "row_count": row_count,
        "output_dimensions": output_dimensions,
        "batch_size": batch_size,
        "workers": workers,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _final_cache_matches(
    settings: Settings,
    *,
    chunks_hash: str,
    model: str,
    dimensions: int | None,
    row_count: int,
) -> bool:
    if not settings.embeddings_path.exists():
        return False
    if not settings.embeddings_hash_path.exists():
        return False

    stored_hash = settings.embeddings_hash_path.read_text(
        encoding="utf-8"
    ).strip()
    if stored_hash != chunks_hash:
        LOGGER.info("embedding_cache_invalid reason=hash_mismatch")
        return False

    metadata = _read_json(_metadata_path(settings))
    if not metadata:
        LOGGER.info("embedding_cache_invalid reason=metadata_missing")
        return False

    if metadata.get("model") != model:
        LOGGER.info("embedding_cache_invalid reason=model_mismatch")
        return False

    if metadata.get("dimensions") != dimensions:
        LOGGER.info("embedding_cache_invalid reason=dimension_mismatch")
        return False

    if int(metadata.get("row_count", -1)) != row_count:
        LOGGER.info("embedding_cache_invalid reason=row_mismatch")
        return False

    embeddings = np.load(settings.embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != row_count:
        LOGGER.info("embedding_cache_invalid reason=file_row_mismatch")
        return False

    output_dimensions = int(metadata.get("output_dimensions", -1))
    if output_dimensions > 0 and embeddings.shape[1] != output_dimensions:
        LOGGER.info("embedding_cache_invalid reason=file_dim_mismatch")
        return False

    return True


def _validate_batch_file(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False

    try:
        embeddings = np.load(path)
    except Exception:
        return False

    return embeddings.ndim == 2 and embeddings.shape[0] == expected_rows


def _prepare_partial_workspace(
    settings: Settings,
    *,
    manifest: dict[str, Any],
    force_rebuild: bool,
) -> None:
    partial_dir = _partial_dir(settings)
    manifest_path = _partial_manifest_path(settings)

    if force_rebuild and partial_dir.exists():
        LOGGER.info("embedding_partial_reset reason=force_rebuild")
        shutil.rmtree(partial_dir)

    if partial_dir.exists():
        existing_manifest = _read_json(manifest_path)
        if existing_manifest != manifest:
            LOGGER.info("embedding_partial_reset reason=manifest_mismatch")
            shutil.rmtree(partial_dir)

    partial_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)


def _save_batch_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, embeddings)
    temp_path.replace(path)


def _get_thread_client(settings: Settings) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "openai_client", None)
    if client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "Falta OPENAI_API_KEY. Define la variable en el entorno, "
                "en .env, o inyéctala antes de lanzar el builder."
            )
        client = OpenAI(api_key=settings.openai_api_key)
        _THREAD_LOCAL.openai_client = client
    return client


def _embed_batch_via_openai(
    settings: Settings,
    texts: Sequence[str],
    *,
    model: str,
    dimensions: int | None,
) -> np.ndarray:
    client = _get_thread_client(settings)
    last_error: Exception | None = None

    for attempt in range(1, settings.openai_max_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "input": list(texts),
            }
            if dimensions is not None:
                request_kwargs["dimensions"] = dimensions

            response = client.embeddings.create(**request_kwargs)
            vectors = [item.embedding for item in response.data]
            return _normalize_embeddings(np.array(vectors, dtype=np.float32))
        except Exception as exc:
            last_error = exc
            if attempt >= settings.openai_max_retries:
                break

            wait_seconds = settings.openai_retry_base_seconds * (
                2 ** (attempt - 1)
            )
            LOGGER.warning(
                "embedding_retry attempt=%s wait_seconds=%.2f error=%s",
                attempt,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"OpenAI embeddings.create failed after retries: {last_error}"
    ) from last_error


def build_embedding_cache(
    settings: Settings,
    texts: Sequence[str],
    *,
    chunks_hash: str,
    model: str = "text-embedding-3-large",
    workers: int = 18,
    batch_size: int = 64,
    dimensions: int | None = None,
    force_rebuild: bool = False,
    embed_batch_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> np.ndarray:
    row_count = len(texts)
    total_batches = (
        (row_count + batch_size - 1) // batch_size if row_count else 0
    )

    if not force_rebuild and _final_cache_matches(
        settings,
        chunks_hash=chunks_hash,
        model=model,
        dimensions=dimensions,
        row_count=row_count,
    ):
        LOGGER.info(
            "embedding_cache_hit path=%s rows=%s",
            settings.embeddings_path,
            row_count,
        )
        return np.load(settings.embeddings_path)

    if row_count == 0:
        empty_embeddings = np.zeros((0, 0), dtype=np.float32)
        np.save(settings.embeddings_path, empty_embeddings)
        settings.embeddings_hash_path.write_text(chunks_hash, encoding="utf-8")
        _write_json(
            _metadata_path(settings),
            _build_metadata(
                chunks_hash=chunks_hash,
                model=model,
                dimensions=dimensions,
                row_count=0,
                output_dimensions=0,
                batch_size=batch_size,
                workers=workers,
            ),
        )
        return empty_embeddings

    resolved_embed_batch_fn = embed_batch_fn
    if resolved_embed_batch_fn is None:
        resolved_embed_batch_fn = lambda batch_texts: _embed_batch_via_openai(
            settings,
            batch_texts,
            model=model,
            dimensions=dimensions,
        )

    manifest = _build_manifest(
        chunks_hash=chunks_hash,
        model=model,
        dimensions=dimensions,
        row_count=row_count,
        batch_size=batch_size,
        total_batches=total_batches,
    )
    _prepare_partial_workspace(
        settings,
        manifest=manifest,
        force_rebuild=force_rebuild,
    )

    batch_specs: list[tuple[int, int, int]] = []
    completed_batches = 0
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, row_count)
        batch_specs.append((batch_index, start, end))

        batch_path = _batch_file_path(settings, batch_index)
        if _validate_batch_file(batch_path, expected_rows=end - start):
            completed_batches += 1
            continue

        if batch_path.exists():
            batch_path.unlink()

    pending_specs = []
    for batch_index, start, end in batch_specs:
        batch_path = _batch_file_path(settings, batch_index)
        if not batch_path.exists():
            pending_specs.append((batch_index, start, end))

    LOGGER.info(
        "embedding_build_start model=%s rows=%s workers=%s batch_size=%s "
        "dimensions=%s completed_batches=%s pending_batches=%s",
        model,
        row_count,
        workers,
        batch_size,
        dimensions,
        completed_batches,
        len(pending_specs),
    )

    if pending_specs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(
                    resolved_embed_batch_fn,
                    list(texts[start:end]),
                ): (batch_index, start, end)
                for batch_index, start, end in pending_specs
            }

            for future in as_completed(future_to_batch):
                batch_index, start, end = future_to_batch[future]
                embeddings = future.result()
                _save_batch_embeddings(
                    _batch_file_path(settings, batch_index), embeddings
                )
                completed_batches += 1
                LOGGER.info(
                    "embedding_batch_complete batch=%s rows=%s completed=%s/%s",
                    batch_index,
                    end - start,
                    completed_batches,
                    total_batches,
                )

    arrays: list[np.ndarray] = []
    output_dimensions = 0
    for batch_index, start, end in batch_specs:
        batch_path = _batch_file_path(settings, batch_index)
        if not _validate_batch_file(batch_path, expected_rows=end - start):
            raise RuntimeError(
                f"Falta o está corrupto el batch {batch_index} en {batch_path}."
            )

        batch_embeddings = np.load(batch_path)
        if output_dimensions == 0:
            output_dimensions = int(batch_embeddings.shape[1])
        elif int(batch_embeddings.shape[1]) != output_dimensions:
            raise RuntimeError(
                "No todos los batches de embeddings tienen la misma dimensión."
            )
        arrays.append(batch_embeddings.astype(np.float32, copy=False))

    final_embeddings = np.vstack(arrays)
    np.save(settings.embeddings_path, final_embeddings)
    settings.embeddings_hash_path.write_text(chunks_hash, encoding="utf-8")
    _write_json(
        _metadata_path(settings),
        _build_metadata(
            chunks_hash=chunks_hash,
            model=model,
            dimensions=dimensions,
            row_count=row_count,
            output_dimensions=output_dimensions,
            batch_size=batch_size,
            workers=workers,
        ),
    )

    shutil.rmtree(_partial_dir(settings), ignore_errors=True)
    LOGGER.info(
        "embedding_build_complete path=%s shape=%s",
        settings.embeddings_path,
        final_embeddings.shape,
    )
    return final_embeddings


def build_row_rag_cache(
    settings: Settings,
    *,
    model: str = "text-embedding-3-large",
    workers: int = 18,
    batch_size: int = 64,
    dimensions: int | None = None,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    catalog_dataframe = load_legacy_excel_catalog(settings)
    chunks_dataframe = build_chunks_df(catalog_dataframe)
    save_chunks_df(chunks_dataframe, settings.chunks_csv_path)

    chunks_hash = compute_chunks_hash(chunks_dataframe)
    embeddings = build_embedding_cache(
        settings,
        chunks_dataframe["text"].tolist(),
        chunks_hash=chunks_hash,
        model=model,
        workers=workers,
        batch_size=batch_size,
        dimensions=dimensions,
        force_rebuild=force_rebuild,
    )
    return chunks_dataframe, embeddings
