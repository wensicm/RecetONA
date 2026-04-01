from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from recetona.config import Settings
from recetona.rag_cache_builder import build_embedding_cache


def _make_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        rag_cache_dir=tmp_path / "rag_cache",
        images_dir=tmp_path / "images",
        catalog_csv_path=tmp_path / "data" / "catalog.csv",
        chunks_csv_path=tmp_path / "rag_cache" / "chunks.csv",
        embeddings_path=tmp_path / "rag_cache" / "embeddings.npy",
        embeddings_hash_path=tmp_path / "rag_cache" / "embeddings.sha256",
        scrape_checkpoint_path=tmp_path / "data" / "scrape_checkpoint.json",
        legacy_excel_path=tmp_path / "mercadona_data.xlsx",
        notebook_path=tmp_path / "mercadona_rag_notebook.ipynb",
    )
    settings.ensure_directories()
    return settings


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def test_build_embedding_cache_resumes_partial_batches(tmp_path: Path):
    settings = _make_settings(tmp_path)
    texts = ["uno", "dos", "tres", "cuatro", "cinco"]
    chunks_hash = "abc123"
    partial_dir = settings.rag_cache_dir / "embeddings.partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "chunks_hash": chunks_hash,
        "model": "text-embedding-3-large",
        "dimensions": None,
        "row_count": len(texts),
        "batch_size": 2,
        "total_batches": 3,
    }
    (partial_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    resumed_batch = _normalize(
        np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    )
    with (partial_dir / "batch_00000.npy").open("wb") as handle:
        np.save(handle, resumed_batch)

    calls: list[list[str]] = []

    def fake_embed(batch_texts):
        calls.append(list(batch_texts))
        matrix = np.array(
            [
                [float(len(text)), float(len(text) + 1), float(len(text) + 2)]
                for text in batch_texts
            ],
            dtype=np.float32,
        )
        return _normalize(matrix)

    embeddings = build_embedding_cache(
        settings,
        texts,
        chunks_hash=chunks_hash,
        model="text-embedding-3-large",
        workers=2,
        batch_size=2,
        embed_batch_fn=fake_embed,
    )

    assert embeddings.shape == (5, 3)
    assert len(calls) == 2
    assert calls == [["tres", "cuatro"], ["cinco"]]
    assert np.allclose(embeddings[:2], resumed_batch)
    assert settings.embeddings_path.exists()
    assert not partial_dir.exists()


def test_build_embedding_cache_reuses_final_cache(tmp_path: Path):
    settings = _make_settings(tmp_path)
    texts = ["aceite", "sal"]
    chunks_hash = "xyz789"
    calls: list[list[str]] = []

    def fake_embed(batch_texts):
        calls.append(list(batch_texts))
        matrix = np.array(
            [
                [float(len(text)), float(len(text) + 1), 1.0]
                for text in batch_texts
            ],
            dtype=np.float32,
        )
        return _normalize(matrix)

    first = build_embedding_cache(
        settings,
        texts,
        chunks_hash=chunks_hash,
        model="text-embedding-3-large",
        workers=1,
        batch_size=2,
        embed_batch_fn=fake_embed,
    )
    assert len(calls) == 1

    second = build_embedding_cache(
        settings,
        texts,
        chunks_hash=chunks_hash,
        model="text-embedding-3-large",
        workers=1,
        batch_size=2,
        embed_batch_fn=lambda _batch: (_ for _ in ()).throw(
            AssertionError("No debería reconstruir el cache final")
        ),
    )

    assert np.allclose(first, second)
