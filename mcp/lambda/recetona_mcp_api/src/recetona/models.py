from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CatalogRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    row_idx: int | None = None
    product_id: str | None = None
    product_name: str = ""
    category: str = ""
    subcategory: str = ""
    subsubcategory: str = ""
    packaging: str = ""
    unit_size: float | None = None
    size_format: str = ""
    price_unit: float | None = None
    price_bulk: float | None = None
    thumbnail_url: str = ""
    photo_urls: str = ""
    ingredients: str = ""
    allergens: str = ""
    nutrition_image_file: str = ""
    nutrition_kj_100: float | None = None
    nutrition_kcal_100: float | None = None
    nutrition_fat_g_100: float | None = None
    nutrition_saturates_g_100: float | None = None
    nutrition_carbs_g_100: float | None = None
    nutrition_sugars_g_100: float | None = None
    nutrition_protein_g_100: float | None = None
    nutrition_salt_g_100: float | None = None
    nutrition_ocr_text: str = ""
    source_updated_at: str = ""


class ChunkRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    row_idx: int
    product_id: str | None = None
    product_name: str = ""
    category: str = ""
    subcategory: str = ""
    subsubcategory: str = ""
    packaging: str = ""
    unit_size: float | None = None
    size_format: str = ""
    price_unit: float | None = None
    price_bulk: float | None = None
    ingredients: str = ""
    allergens: str = ""
    nutrition_ocr_text: str = ""
    text: str
    lexical_text: str
    ingredient_search_text: str
    ingredient_desc_text: str


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    url: str


class FetchPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecipeAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str
    block_1: str = ""
    block_2: str = ""
    block_3: str = ""
    inferred_ingredients: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    cost_summary: dict[str, Any] = Field(default_factory=dict)


CATALOG_COLUMNS = list(CatalogRow.model_fields)
CHUNK_COLUMNS = list(ChunkRow.model_fields)
