from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .models import CATALOG_COLUMNS
from .utils import clean_text, normalize_product_id, safe_float


LOGGER = logging.getLogger(__name__)


LEGACY_EXCEL_MAPPING = {
    "product_id": "product_id",
    "product_name": "product_name",
    "category": "category",
    "subcategory": "subcategory",
    "subsubcategory": "subsubcategory",
    "packaging": "packaging",
    "unit_size": "unit_size",
    "size_format": "size_format",
    "price_unit": "price_unit",
    "price_bulk": "price_bulk",
    "thumbnail_url": "thumbnail_url",
    "photo_urls": "photo_urls",
    "Ingredientes": "ingredients",
    "Alérgenos": "allergens",
    "nutrition_image_file": "nutrition_image_file",
    "nutrition_kj_100": "nutrition_kj_100",
    "nutrition_kcal_100": "nutrition_kcal_100",
    "nutrition_fat_g_100": "nutrition_fat_g_100",
    "nutrition_saturates_g_100": "nutrition_saturates_g_100",
    "nutrition_carbs_g_100": "nutrition_carbs_g_100",
    "nutrition_sugars_g_100": "nutrition_sugars_g_100",
    "nutrition_protein_g_100": "nutrition_protein_g_100",
    "nutrition_salt_g_100": "nutrition_salt_g_100",
    "nutrition_ocr_text": "nutrition_ocr_text",
}


LEGACY_CHUNKS_MAPPING = {
    "row_idx": "row_idx",
    "product_id": "product_id",
    "product_name": "product_name",
    "category": "category",
    "subcategory": "subcategory",
    "subsubcategory": "subsubcategory",
    "packaging": "packaging",
    "unit_size": "unit_size",
    "size_format": "size_format",
    "price_unit": "price_unit",
    "price_bulk": "price_bulk",
    "ingredientes": "ingredients",
    "alergenos": "allergens",
    "nutrition_ocr_text": "nutrition_ocr_text",
}


TEXT_COLUMNS = {
    "product_name",
    "category",
    "subcategory",
    "subsubcategory",
    "packaging",
    "size_format",
    "thumbnail_url",
    "photo_urls",
    "ingredients",
    "allergens",
    "nutrition_image_file",
    "nutrition_ocr_text",
    "source_updated_at",
}


FLOAT_COLUMNS = {
    "unit_size",
    "price_unit",
    "price_bulk",
    "nutrition_kj_100",
    "nutrition_kcal_100",
    "nutrition_fat_g_100",
    "nutrition_saturates_g_100",
    "nutrition_carbs_g_100",
    "nutrition_sugars_g_100",
    "nutrition_protein_g_100",
    "nutrition_salt_g_100",
}


def empty_catalog_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CATALOG_COLUMNS)


def ensure_catalog_schema(df: pd.DataFrame) -> pd.DataFrame:
    catalog = df.copy()
    for column in CATALOG_COLUMNS:
        if column not in catalog.columns:
            catalog[column] = "" if column in TEXT_COLUMNS else None

    if "row_idx" not in catalog.columns or catalog["row_idx"].isna().all():
        catalog["row_idx"] = np.arange(len(catalog), dtype=int)

    catalog["row_idx"] = pd.Series(range(len(catalog)), dtype=int)
    catalog["product_id"] = catalog["product_id"].apply(normalize_product_id)

    for column in TEXT_COLUMNS:
        catalog[column] = catalog[column].apply(clean_text)

    for column in FLOAT_COLUMNS:
        catalog[column] = catalog[column].apply(safe_float)

    catalog = catalog[CATALOG_COLUMNS]
    return catalog


def _load_legacy_excel(path: Path) -> pd.DataFrame:
    legacy = pd.read_excel(path)
    data = {}
    for source_column, target_column in LEGACY_EXCEL_MAPPING.items():
        if source_column in legacy.columns:
            data[target_column] = legacy[source_column]
    migrated = pd.DataFrame(data)
    migrated["source_updated_at"] = ""
    return ensure_catalog_schema(migrated)


def _load_legacy_chunks(path: Path) -> pd.DataFrame:
    legacy = pd.read_csv(path)
    data = {}
    for source_column, target_column in LEGACY_CHUNKS_MAPPING.items():
        if source_column in legacy.columns:
            data[target_column] = legacy[source_column]
    migrated = pd.DataFrame(data)
    if "thumbnail_url" not in migrated.columns:
        migrated["thumbnail_url"] = ""
    if "photo_urls" not in migrated.columns:
        migrated["photo_urls"] = ""
    migrated["source_updated_at"] = ""
    return ensure_catalog_schema(migrated)


def save_catalog_df(df: pd.DataFrame, path: Path) -> Path:
    catalog = ensure_catalog_schema(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    return path


def bootstrap_catalog_from_legacy(settings: Settings) -> Path:
    if settings.catalog_csv_path.exists():
        return settings.catalog_csv_path

    if settings.legacy_excel_path.exists():
        LOGGER.info("catalog_bootstrap source=excel path=%s", settings.legacy_excel_path)
        save_catalog_df(_load_legacy_excel(settings.legacy_excel_path), settings.catalog_csv_path)
        return settings.catalog_csv_path

    if settings.chunks_csv_path.exists():
        LOGGER.info("catalog_bootstrap source=chunks path=%s", settings.chunks_csv_path)
        save_catalog_df(_load_legacy_chunks(settings.chunks_csv_path), settings.catalog_csv_path)
        return settings.catalog_csv_path

    raise FileNotFoundError(
        f"No existe catalogo en {settings.catalog_csv_path}, ni legacy Excel en {settings.legacy_excel_path}, "
        f"ni chunks en {settings.chunks_csv_path}."
    )


def load_catalog_df(settings: Settings, allow_bootstrap: bool = True) -> pd.DataFrame:
    path = settings.catalog_csv_path
    if not path.exists():
        if not allow_bootstrap:
            raise FileNotFoundError(f"No existe catalogo canonico en {path}.")
        bootstrap_catalog_from_legacy(settings)
    catalog = pd.read_csv(path)
    return ensure_catalog_schema(catalog)


def export_catalog_to_excel(catalog_df: pd.DataFrame, output_path: Path) -> Path:
    catalog = ensure_catalog_schema(catalog_df)
    export_df = catalog.rename(
        columns={
            "ingredients": "Ingredientes",
            "allergens": "Alérgenos",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(output_path, index=False)
    return output_path


def row_to_fetch_payload(row: pd.Series) -> dict[str, Any]:
    product_id = normalize_product_id(row.get("product_id")) or str(row.get("row_idx", "")).strip()
    title = clean_text(row.get("product_name")) or f"Producto {product_id}"
    text = (
        f"Producto: {title}\n"
        f"Categoria: {clean_text(row.get('category'))}\n"
        f"Subcategoria: {clean_text(row.get('subcategory'))}\n"
        f"Precio unidad: {clean_text(row.get('price_unit'))}\n"
        f"Ingredientes: {clean_text(row.get('ingredients'))}"
    ).strip()
    return {
        "id": product_id,
        "title": title,
        "text": text,
        "url": f"recetona://producto/{product_id}",
        "metadata": {
            "category": clean_text(row.get("category")),
            "subcategory": clean_text(row.get("subcategory")),
            "subsubcategory": clean_text(row.get("subsubcategory")),
            "price_unit": row.get("price_unit"),
            "unit_size": row.get("unit_size"),
            "size_format": clean_text(row.get("size_format")),
        },
    }
