from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import load_catalog_df
from .config import Settings
from .llm import OpenAIBackend
from .models import CHUNK_COLUMNS
from .utils import clean_text, numeric_text

LOGGER = logging.getLogger(__name__)


def build_row_text(row: pd.Series) -> str:
    parts = [
        f"Producto: {clean_text(row.get('product_name'))}",
        f"ID producto: {numeric_text(row.get('product_id'))}",
        f"Categoria: {clean_text(row.get('category'))} > {clean_text(row.get('subcategory'))} > {clean_text(row.get('subsubcategory'))}",
        f"Formato: {clean_text(row.get('packaging'))}, {numeric_text(row.get('unit_size'))} {clean_text(row.get('size_format'))}",
        f"Precio unidad: {numeric_text(row.get('price_unit'))} EUR",
        f"Precio por volumen/peso: {numeric_text(row.get('price_bulk'))} EUR",
        f"Ingredientes: {clean_text(row.get('ingredients'))}",
        f"Alergenos: {clean_text(row.get('allergens'))}",
        (
            "Nutricion por 100g/ml: "
            f"kJ={numeric_text(row.get('nutrition_kj_100'))}; "
            f"kcal={numeric_text(row.get('nutrition_kcal_100'))}; "
            f"grasas={numeric_text(row.get('nutrition_fat_g_100'))} g; "
            f"saturadas={numeric_text(row.get('nutrition_saturates_g_100'))} g; "
            f"hidratos={numeric_text(row.get('nutrition_carbs_g_100'))} g; "
            f"azucares={numeric_text(row.get('nutrition_sugars_g_100'))} g; "
            f"proteinas={numeric_text(row.get('nutrition_protein_g_100'))} g; "
            f"sal={numeric_text(row.get('nutrition_salt_g_100'))} g"
        ),
        f"Imagen principal: {clean_text(row.get('thumbnail_url'))}",
        f"Imagenes: {clean_text(row.get('photo_urls'))}",
    ]
    return "\n".join(parts)


def build_chunks_df(catalog_df: pd.DataFrame) -> pd.DataFrame:
    chunks = pd.DataFrame(
        {
            "row_idx": catalog_df["row_idx"],
            "product_id": catalog_df["product_id"],
            "product_name": catalog_df["product_name"],
            "category": catalog_df["category"],
            "subcategory": catalog_df["subcategory"],
            "subsubcategory": catalog_df["subsubcategory"],
            "packaging": catalog_df["packaging"],
            "unit_size": catalog_df["unit_size"],
            "size_format": catalog_df["size_format"],
            "price_unit": catalog_df["price_unit"],
            "price_bulk": catalog_df["price_bulk"],
            "ingredients": catalog_df["ingredients"],
            "allergens": catalog_df["allergens"],
            "nutrition_ocr_text": catalog_df["nutrition_ocr_text"],
            "text": [build_row_text(row) for _, row in catalog_df.iterrows()],
        }
    )

    chunks["lexical_text"] = (
        chunks["product_name"].fillna("").astype(str)
        + " "
        + chunks["category"].fillna("").astype(str)
        + " "
        + chunks["subcategory"].fillna("").astype(str)
        + " "
        + chunks["subsubcategory"].fillna("").astype(str)
        + " "
        + chunks["ingredients"].fillna("").astype(str)
    )
    chunks["ingredient_search_text"] = (
        chunks["product_name"].fillna("").astype(str)
        + " "
        + chunks["category"].fillna("").astype(str)
        + " "
        + chunks["subcategory"].fillna("").astype(str)
        + " "
        + chunks["subsubcategory"].fillna("").astype(str)
    )
    chunks["ingredient_desc_text"] = (
        chunks["ingredients"].fillna("").astype(str)
        + " "
        + chunks["allergens"].fillna("").astype(str)
        + " "
        + chunks["nutrition_ocr_text"]
        .fillna("")
        .astype(str)
        .str.slice(0, 2500)
    )
    return chunks[CHUNK_COLUMNS]


def compute_chunks_hash(chunks_df: pd.DataFrame) -> str:
    payload = chunks_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_chunks_df(chunks_df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(path, index=False)
    return path


def load_chunks_df(
    settings: Settings, allow_build: bool = True
) -> pd.DataFrame:
    if settings.chunks_csv_path.exists():
        return pd.read_csv(settings.chunks_csv_path)
    if not allow_build:
        raise FileNotFoundError(
            f"No existe el indice de chunks en {settings.chunks_csv_path}."
        )
    catalog_df = load_catalog_df(settings)
    chunks_df = build_chunks_df(catalog_df)
    save_chunks_df(chunks_df, settings.chunks_csv_path)
    return chunks_df


def load_embeddings(
    settings: Settings, chunks_df: pd.DataFrame
) -> np.ndarray | None:
    if (
        not settings.embeddings_path.exists()
        or not settings.embeddings_hash_path.exists()
    ):
        return None
    expected_hash = compute_chunks_hash(chunks_df)
    stored_hash = settings.embeddings_hash_path.read_text(
        encoding="utf-8"
    ).strip()
    if stored_hash != expected_hash:
        LOGGER.info("embedding_cache_invalid reason=hash_mismatch")
        return None
    embeddings = np.load(settings.embeddings_path)
    if embeddings.shape[0] != len(chunks_df):
        LOGGER.info("embedding_cache_invalid reason=row_mismatch")
        return None
    return embeddings


def ensure_embeddings(
    settings: Settings,
    backend: OpenAIBackend,
    chunks_df: pd.DataFrame,
    force_rebuild: bool = False,
) -> np.ndarray:
    if not force_rebuild:
        cached = load_embeddings(settings, chunks_df)
        if cached is not None:
            return cached

    embeddings = backend.embed_texts(
        chunks_df["text"].tolist(), model=settings.embed_model
    )
    settings.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(settings.embeddings_path, embeddings)
    settings.embeddings_hash_path.write_text(
        compute_chunks_hash(chunks_df), encoding="utf-8"
    )
    return embeddings


def rebuild_index(
    settings: Settings,
    backend: OpenAIBackend | None = None,
    force_embeddings: bool = False,
) -> pd.DataFrame:
    catalog_df = load_catalog_df(settings)
    chunks_df = build_chunks_df(catalog_df)
    save_chunks_df(chunks_df, settings.chunks_csv_path)
    if backend and backend.enabled:
        ensure_embeddings(
            settings, backend, chunks_df, force_rebuild=force_embeddings
        )
    return chunks_df
