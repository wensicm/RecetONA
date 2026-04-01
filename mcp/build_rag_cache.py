#!/usr/bin/env python3.12
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recetona.config import Settings, configure_logging
from recetona.rag_cache_builder import build_row_rag_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera rag_cache/chunks.csv y rag_cache/embeddings.npy "
            "a partir de mercadona_data.xlsx."
        )
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="Modelo de embeddings de OpenAI.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=18,
        help="Número de workers concurrentes para embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de lote por request a embeddings.create.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help=(
            "Dimensión de salida opcional. Si se omite, el modelo usa su "
            "dimensión nativa."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinicia los parciales y regenera embeddings desde cero.",
    )
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    chunks_dataframe, embeddings = build_row_rag_cache(
        settings,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
        force_rebuild=args.force,
    )

    print(f"Chunks: {settings.chunks_csv_path}")
    print(f"Embeddings: {settings.embeddings_path}")
    print(f"Filas indexadas: {len(chunks_dataframe)}")
    print(f"Shape embeddings: {embeddings.shape}")


if __name__ == "__main__":
    main()
