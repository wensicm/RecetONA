from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag_server import (
    ENV_PATH,
    NOTEBOOK_PATH,
    NotebookRagService,
    _candidate_is_incompatible_for_ingredient,
    _normalize_matching_text,
    ensure_runtime_rag_cache,
    load_env_file,
)

MCP_NAME = "RecetONA"
APP_SERVICE_NAME = "recetona-chatgpt-app"
RECIPE_WIDGET_URI = "ui://widget/recetona-recipe-v2.html"
CATALOG_CACHE_PATH = PROJECT_ROOT / "rag_cache" / "chunks.csv"
EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "rag_cache" / "embeddings.npy"
EXCEL_PATH = PROJECT_ROOT / "mercadona_data.xlsx"

_service: NotebookRagService | None = None
_catalog_dataframe: pd.DataFrame | None = None
_resolved_openai_api_key: str | None = None
_resolved_openai_api_key_source: str | None = None
FALLBACK_INGREDIENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "chocolate negro",
        re.compile(r"\bchocolate(?: negro)?\b", re.IGNORECASE),
    ),
    ("cacao puro", re.compile(r"\bcacao\b", re.IGNORECASE)),
    ("mantequilla", re.compile(r"\bmantequilla\b", re.IGNORECASE)),
    ("azucar", re.compile(r"\baz[uú]car\b", re.IGNORECASE)),
    ("huevos", re.compile(r"\bhuev(?:o|os)\b", re.IGNORECASE)),
    ("harina", re.compile(r"\bharina\b", re.IGNORECASE)),
    ("leche", re.compile(r"\bleche\b", re.IGNORECASE)),
    ("sal", re.compile(r"\bsal\b", re.IGNORECASE)),
    ("vainilla", re.compile(r"\bvainilla\b", re.IGNORECASE)),
    (
        "gasificante reposteria",
        re.compile(
            r"\b(?:gasificante|levadura|polvo de hornear)\b",
            re.IGNORECASE,
        ),
    ),
    ("canela", re.compile(r"\bcanela\b", re.IGNORECASE)),
    ("gelatina", re.compile(r"\bgelatina\b", re.IGNORECASE)),
    ("nueces", re.compile(r"\bnue(?:z|ces)\b", re.IGNORECASE)),
)
FALLBACK_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "azucar": ("azucar", "azúcar"),
    "cacao puro": ("cacao", "puro"),
    "canela": ("canela",),
    "gelatina": ("gelatina",),
    "chocolate negro": ("chocolate", "negro", "fundir"),
    "gasificante reposteria": ("gasificante", "reposteria"),
    "harina": ("harina",),
    "huevos": ("huevos",),
    "leche": ("leche",),
    "mantequilla": ("mantequilla",),
    "nueces": ("nueces", "nuez"),
    "sal": ("sal",),
    "vainilla": ("vainilla",),
}


class RecipeIngredientItem(BaseModel):
    ingrediente_objetivo: str | None = None
    producto_mercadona: str
    producto_id: str | None = None
    cantidad_receta_valor: float | None = None
    cantidad_receta_unidad: str | None = None
    tamano_envase_valor: float | None = None
    tamano_envase_unidad: str | None = None
    precio_envase_eur: float | None = None
    envases_a_comprar: float | None = None
    coste_compra_eur: float | None = None
    coste_consumido_eur: float | None = None
    notas: str | None = None


class RecipeWidgetPayload(BaseModel):
    pregunta: str
    productos_mercadona_exactos: list[str] = Field(default_factory=list)
    ingredientes_mercadona: list[RecipeIngredientItem] = Field(
        default_factory=list
    )
    ingredientes_mercadona_texto_literal: str = ""
    receta_y_pasos_texto_literal: str = ""
    respuesta_literal_mcp: str = ""
    coste_total_compra_eur: float | None = None
    coste_total_consumido_eur: float | None = None
    personas: int | None = None
    ingredientes_inferidos: list[str] = Field(default_factory=list)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def _resolve_recipe_widget_html_path() -> Path:
    candidate_paths = (
        PROJECT_ROOT.parent / "web" / "public" / "recetona-widget.html",
        PROJECT_ROOT / "web" / "public" / "recetona-widget.html",
    )
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    joined_candidates = ", ".join(str(path) for path in candidate_paths)
    raise RuntimeError(
        "No se encontro el widget HTML de RecetONA. Rutas probadas: "
        f"{joined_candidates}"
    )


def _get_widget_domain() -> str | None:
    raw_value = os.getenv("RECETONA_WIDGET_DOMAIN", "").strip()
    if not raw_value:
        return None
    return raw_value.rstrip("/")


def _format_product_id(value: Any) -> str:
    try:
        numeric_value = float(value)
    except Exception:
        return str(value).strip()
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"


def _env_flag_is_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _require_prebuilt_rag_cache() -> bool:
    return _env_flag_is_enabled("RECETONA_REQUIRE_PREBUILT_RAG_CACHE")


def _get_openai_api_key_ssm_parameter_name() -> str:
    return os.getenv("OPENAI_API_KEY_SSM_PARAMETER", "").strip()


def _resolve_openai_api_key(*, raise_on_error: bool) -> str | None:
    global _resolved_openai_api_key
    global _resolved_openai_api_key_source

    if _resolved_openai_api_key:
        return _resolved_openai_api_key

    direct_value = os.getenv("OPENAI_API_KEY", "").strip()
    if direct_value:
        _resolved_openai_api_key = direct_value
        _resolved_openai_api_key_source = "env"
        return _resolved_openai_api_key

    parameter_name = _get_openai_api_key_ssm_parameter_name()
    if not parameter_name:
        return None

    try:
        import boto3

        region_name = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "eu-west-1"
        )
        ssm_client = boto3.client("ssm", region_name=region_name)
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True,
        )
        parameter_value = (
            response.get("Parameter", {}).get("Value", "").strip()
        )
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(
                "No se pudo leer OPENAI_API_KEY desde SSM Parameter Store "
                f"('{parameter_name}')."
            ) from exc
        return None

    if not parameter_value:
        return None

    os.environ["OPENAI_API_KEY"] = parameter_value
    _resolved_openai_api_key = parameter_value
    _resolved_openai_api_key_source = "ssm"
    return _resolved_openai_api_key


def _recipe_runtime_status() -> dict[str, Any]:
    missing_assets: list[str] = []
    parameter_name = _get_openai_api_key_ssm_parameter_name()
    openai_api_key = _resolve_openai_api_key(raise_on_error=False)
    cache_status = ensure_runtime_rag_cache(
        raise_on_error=False,
        eager_download=True,
    )

    if not EXCEL_PATH.exists():
        missing_assets.append(str(EXCEL_PATH.name))

    if _require_prebuilt_rag_cache():
        missing_assets.extend(
            [
                str(cache_status["cache_dir"] / filename)
                for filename in cache_status["missing_files"]
            ]
        )

    return {
        "openai_api_key_present": bool(openai_api_key),
        "openai_api_key_source": (
            _resolved_openai_api_key_source if openai_api_key else "missing"
        ),
        "openai_api_key_ssm_parameter": parameter_name or None,
        "requires_prebuilt_rag_cache": _require_prebuilt_rag_cache(),
        "cache_ready": not cache_status["missing_files"],
        "cache_dir": str(cache_status["cache_dir"]),
        "cache_remote_configured": cache_status["remote_configured"],
        "cache_s3_bucket": cache_status["s3_bucket"],
        "cache_s3_prefix": cache_status["s3_prefix"],
        "missing_assets": missing_assets,
    }


def _validate_recipe_runtime_assets() -> None:
    status = _recipe_runtime_status()
    if not status["requires_prebuilt_rag_cache"]:
        return

    missing_assets = status["missing_assets"]
    if not missing_assets:
        return

    joined_assets = ", ".join(missing_assets)
    raise RuntimeError(
        "query_recipe_data requiere caches RAG precalculadas en Lambda. "
        "Faltan los artefactos: "
        f"{joined_assets}. "
        "Genera rag_cache/chunks.csv y rag_cache/embeddings.npy antes "
        "de desplegar o desactiva RECETONA_REQUIRE_PREBUILT_RAG_CACHE."
    )


def _get_service() -> NotebookRagService:
    global _service
    if _service is not None:
        return _service

    load_env_file(ENV_PATH)
    if not _resolve_openai_api_key(raise_on_error=True):
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Define la clave en variables de entorno "
            f"o en {ENV_PATH}, o configura OPENAI_API_KEY_SSM_PARAMETER."
        )

    _validate_recipe_runtime_assets()

    _service = NotebookRagService(NOTEBOOK_PATH)
    return _service


def _load_catalog() -> pd.DataFrame:
    global _catalog_dataframe
    if _catalog_dataframe is not None:
        return _catalog_dataframe

    if CATALOG_CACHE_PATH.exists():
        catalog_dataframe = pd.read_csv(CATALOG_CACHE_PATH)
    elif EXCEL_PATH.exists():
        catalog_dataframe = pd.read_excel(EXCEL_PATH)
        catalog_dataframe["row_idx"] = np.arange(
            len(catalog_dataframe), dtype=int
        )
    else:
        raise RuntimeError(
            f"No existe catalogo en {CATALOG_CACHE_PATH} ni {EXCEL_PATH}."
        )

    for column_name in (
        "product_name",
        "category",
        "subcategory",
        "subsubcategory",
        "ingredientes",
    ):
        if column_name not in catalog_dataframe.columns:
            catalog_dataframe[column_name] = ""

    catalog_dataframe["search_text"] = (
        catalog_dataframe["product_name"].fillna("").astype(str)
        + " "
        + catalog_dataframe["category"].fillna("").astype(str)
        + " "
        + catalog_dataframe["subcategory"].fillna("").astype(str)
        + " "
        + catalog_dataframe["subsubcategory"].fillna("").astype(str)
        + " "
        + catalog_dataframe["ingredientes"].fillna("").astype(str)
    ).str.lower()

    _catalog_dataframe = catalog_dataframe
    return _catalog_dataframe


def _row_to_fetch_payload(row: pd.Series) -> dict[str, Any]:
    product_id = _format_product_id(
        row.get("product_id", row.get("row_idx", ""))
    )
    title = str(row.get("product_name", f"Producto {product_id}")).strip()
    text = str(row.get("text", "")).strip()
    if not text:
        text = (
            f"Producto: {title}\n"
            f"Categoria: {row.get('category', '')}\n"
            f"Subcategoria: {row.get('subcategory', '')}\n"
            f"Precio unidad: {row.get('price_unit', '')}\n"
            f"Ingredientes: {row.get('ingredientes', '')}"
        ).strip()

    return {
        "id": product_id,
        "title": title,
        "text": text,
        "url": f"recetona://producto/{product_id}",
        "metadata": {
            "category": row.get("category", ""),
            "subcategory": row.get("subcategory", ""),
            "subsubcategory": row.get("subsubcategory", ""),
            "price_unit": row.get("price_unit", ""),
            "unit_size": row.get("unit_size", ""),
            "size_format": row.get("size_format", ""),
        },
    }


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _normalize_cost_plan_rows(cost_plan: Any) -> list[dict[str, Any]]:
    if isinstance(cost_plan, pd.DataFrame):
        rows = cost_plan.to_dict(orient="records")
    elif isinstance(cost_plan, list):
        rows = cost_plan
    else:
        return []

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        product_name = str(row.get("product_name", "")).strip()
        if not product_name or product_name.lower() == "nan":
            continue

        product_id_raw = row.get("product_id")
        product_id = None
        if product_id_raw is not None and str(product_id_raw).strip():
            formatted_id = _format_product_id(product_id_raw)
            product_id = formatted_id or None

        normalized_rows.append(
            {
                "ingrediente_objetivo": str(row.get("ingredient", "")).strip(),
                "producto_mercadona": product_name,
                "producto_id": product_id,
                "cantidad_receta_valor": _safe_float_or_none(
                    row.get("required_qty")
                ),
                "cantidad_receta_unidad": str(
                    row.get("required_unit", "")
                ).strip()
                or None,
                "tamano_envase_valor": _safe_float_or_none(
                    row.get("unit_size")
                ),
                "tamano_envase_unidad": str(row.get("size_format", "")).strip()
                or None,
                "precio_envase_eur": _safe_float_or_none(
                    row.get("price_unit")
                ),
                "envases_a_comprar": _safe_float_or_none(
                    row.get("units_to_buy")
                ),
                "coste_compra_eur": _safe_float_or_none(
                    row.get("purchase_cost_eur")
                ),
                "coste_consumido_eur": _safe_float_or_none(
                    row.get("escandallo_cost_eur")
                ),
                "notas": str(row.get("notes", "")).strip() or None,
            }
        )

    return normalized_rows


def _build_query_recipe_payload(
    *,
    pregunta: str,
    result: dict[str, Any],
) -> RecipeWidgetPayload:
    ingredients = _normalize_cost_plan_rows(result.get("cost_plan"))
    cost_summary = result.get("cost_summary")
    if not isinstance(cost_summary, dict):
        cost_summary = {}

    inferred_ingredients = result.get("inferred_ingredients") or []

    if not ingredients:
        fallback_ingredients = _build_catalog_fallback_ingredients(
            pregunta=pregunta,
            result=result,
        )
        if fallback_ingredients:
            ingredients = [
                ingredient.model_dump(mode="json")
                for ingredient in fallback_ingredients
            ]
            if not inferred_ingredients:
                inferred_ingredients = [
                    ingredient.ingrediente_objetivo or ""
                    for ingredient in fallback_ingredients
                    if ingredient.ingrediente_objetivo
                ]
            if not cost_summary.get("total_purchase_eur"):
                cost_summary["total_purchase_eur"] = sum(
                    ingredient.coste_compra_eur or 0.0
                    for ingredient in fallback_ingredients
                )

    ingredientes_texto_literal = str(result.get("block_1", "")).strip()
    if ingredients and (
        not ingredientes_texto_literal
        or "No se encontraron ingredientes/productos"
        in ingredientes_texto_literal
    ):
        ingredientes_texto_literal = _build_ingredient_lines_text(ingredients)

    respuesta_literal_mcp = str(result.get("answer", "")).strip()
    if (
        ingredients
        and "No se encontraron ingredientes/productos" in respuesta_literal_mcp
    ):
        receta_texto = str(result.get("block_3", "")).strip()
        respuesta_literal_mcp = (
            "Ingredientes de Mercadona:\n"
            f"{ingredientes_texto_literal}\n\n"
            "Receta y pasos:\n\n"
            f"{receta_texto}"
        ).strip()

    return RecipeWidgetPayload(
        pregunta=pregunta,
        productos_mercadona_exactos=[
            item["producto_mercadona"] for item in ingredients
        ],
        ingredientes_mercadona=ingredients,
        ingredientes_mercadona_texto_literal=ingredientes_texto_literal,
        receta_y_pasos_texto_literal=str(result.get("block_3", "")).strip(),
        respuesta_literal_mcp=respuesta_literal_mcp,
        coste_total_compra_eur=_safe_float_or_none(
            cost_summary.get("total_purchase_eur")
        ),
        coste_total_consumido_eur=_safe_float_or_none(
            cost_summary.get("total_escandallo_eur")
        ),
        personas=cost_summary.get("servings"),
        ingredientes_inferidos=inferred_ingredients,
    )


def _extract_fallback_ingredients(
    *,
    pregunta: str,
    result: dict[str, Any],
) -> list[str]:
    ingredients: list[str] = []
    seen: set[str] = set()

    for item in result.get("inferred_ingredients") or []:
        normalized_item = _normalize_matching_text(item)
        if normalized_item and normalized_item not in seen:
            seen.add(normalized_item)
            ingredients.append(normalized_item)

    recipe_text = "\n".join(
        str(result.get(key, "")).strip()
        for key in ("block_1", "block_3", "answer")
    )
    haystack = f"{pregunta}\n{recipe_text}"

    for ingredient_name, pattern in FALLBACK_INGREDIENT_PATTERNS:
        if ingredient_name in seen:
            continue
        if pattern.search(haystack):
            seen.add(ingredient_name)
            ingredients.append(ingredient_name)

    return ingredients


def _score_catalog_row_for_ingredient(
    *,
    ingredient: str,
    row: pd.Series,
) -> int:
    product_name = _normalize_matching_text(row.get("product_name"))
    category = _normalize_matching_text(row.get("category"))
    subcategory = _normalize_matching_text(row.get("subcategory"))
    subsubcategory = _normalize_matching_text(row.get("subsubcategory"))
    haystack = " ".join(
        part
        for part in (product_name, category, subcategory, subsubcategory)
        if part
    )
    search_terms = FALLBACK_SEARCH_TERMS.get(ingredient, (ingredient,))
    score = 0
    for term in search_terms:
        normalized_term = _normalize_matching_text(term)
        if normalized_term and normalized_term in product_name:
            score += 4
        elif normalized_term and normalized_term in haystack:
            score += 2

    if ingredient == "chocolate negro" and "fundir" in product_name:
        score += 2
    if ingredient == "gasificante reposteria" and "reposteria" in haystack:
        score += 2
    if ingredient == "huevos" and "huevos" in product_name:
        score += 2

    return score


def _row_is_compatible_for_fallback_ingredient(
    *,
    ingredient: str,
    row: pd.Series,
) -> bool:
    product_name = _normalize_matching_text(row.get("product_name"))
    category = _normalize_matching_text(row.get("category"))
    subcategory = _normalize_matching_text(row.get("subcategory"))
    subsubcategory = _normalize_matching_text(row.get("subsubcategory"))
    taxonomy = " ".join(
        part for part in (category, subcategory, subsubcategory) if part
    )

    if ingredient == "azucar":
        return "azucar" in product_name and "azucar" in taxonomy
    if ingredient == "cacao puro":
        return "cacao" in product_name
    if ingredient == "chocolate negro":
        return "chocolate" in product_name and (
            "negro" in product_name or "fundir" in product_name
        )
    if ingredient == "gasificante reposteria":
        return (
            "gasificante" in product_name or "levadura" in product_name
        ) and "reposteria" in taxonomy + " " + product_name
    if ingredient == "harina":
        return "harina" in product_name and "harina" in taxonomy
    if ingredient == "huevos":
        return "huevos" in product_name
    if ingredient == "leche":
        return "leche" in product_name
    if ingredient == "mantequilla":
        return "mantequilla" in product_name and (
            "mantequilla" in taxonomy or "margarina" in taxonomy
        )
    if ingredient == "sal":
        return "sal" in product_name and "sal" in taxonomy
    if ingredient == "vainilla":
        return "vainilla" in product_name
    if ingredient == "canela":
        return "canela" in product_name
    if ingredient == "gelatina":
        return "gelatina" in product_name
    if ingredient == "nueces":
        return "nuez" in product_name or "nueces" in product_name

    return True


def _build_fallback_ingredient_item(
    *,
    ingredient: str,
    row: pd.Series,
) -> RecipeIngredientItem:
    price_unit = _safe_float_or_none(row.get("price_unit"))
    return RecipeIngredientItem(
        ingrediente_objetivo=ingredient,
        producto_mercadona=str(row.get("product_name", "")).strip(),
        producto_id=_format_product_id(
            row.get("product_id", row.get("row_idx", ""))
        )
        or None,
        cantidad_receta_valor=1.0,
        cantidad_receta_unidad="ud",
        tamano_envase_valor=_safe_float_or_none(row.get("unit_size")),
        tamano_envase_unidad=str(row.get("size_format", "")).strip() or None,
        precio_envase_eur=price_unit,
        envases_a_comprar=1.0,
        coste_compra_eur=price_unit,
        coste_consumido_eur=None,
        notas="Fallback desde catalogo por ausencia de cost_plan",
    )


def _build_catalog_fallback_ingredients(
    *,
    pregunta: str,
    result: dict[str, Any],
) -> list[RecipeIngredientItem]:
    fallback_ingredients = _extract_fallback_ingredients(
        pregunta=pregunta,
        result=result,
    )
    if not fallback_ingredients:
        return []

    catalog_dataframe = _load_catalog()
    selected_products: set[str] = set()
    selected_items: list[RecipeIngredientItem] = []

    for ingredient in fallback_ingredients:
        scored_candidates: list[tuple[int, float, pd.Series]] = []
        for _, row in catalog_dataframe.iterrows():
            if _candidate_is_incompatible_for_ingredient(
                ingredient,
                row,
                recipe_query=pregunta,
            ):
                continue
            if not _row_is_compatible_for_fallback_ingredient(
                ingredient=ingredient,
                row=row,
            ):
                continue
            score = _score_catalog_row_for_ingredient(
                ingredient=ingredient,
                row=row,
            )
            if score <= 0:
                continue
            price_unit = _safe_float_or_none(row.get("price_unit"))
            scored_candidates.append(
                (
                    score,
                    price_unit if price_unit is not None else 10_000.0,
                    row,
                )
            )

        if not scored_candidates:
            continue

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        best_row = next(
            (
                row
                for _, _, row in scored_candidates
                if str(row.get("product_name", "")).strip()
                not in selected_products
            ),
            scored_candidates[0][2],
        )
        product_name = str(best_row.get("product_name", "")).strip()
        selected_products.add(product_name)
        selected_items.append(
            _build_fallback_ingredient_item(
                ingredient=ingredient,
                row=best_row,
            )
        )

    return selected_items


def _format_eur(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.2f}".replace(".", ",") + "€"


def _build_ingredient_lines_text(ingredients: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for ingredient in ingredients:
        product_name = str(ingredient.get("producto_mercadona", "")).strip()
        if not product_name:
            continue
        price_text = _format_eur(
            _safe_float_or_none(ingredient.get("precio_envase_eur"))
        )
        line = f"- 1 ud {product_name}"
        if price_text:
            line += f" ({price_text})"
        lines.append(line)
    return "\n".join(lines)


def _recipe_widget_resource_meta() -> dict[str, Any]:
    meta = {
        "ui": {
            "prefersBorder": True,
            "csp": {
                "connectDomains": [],
                "resourceDomains": [],
            },
        },
        "openai/widgetDescription": (
            "Renderiza la receta de RecetONA con los nombres exactos de los "
            "productos de Mercadona, precios y pasos. El widget ya muestra "
            "la lista exacta, asi que evita repetirla o parafrasearla fuera "
            "del panel."
        ),
        "openai/widgetPrefersBorder": True,
        "openai/widgetCSP": {
            "connect_domains": [],
            "resource_domains": [],
        },
    }
    widget_domain = _get_widget_domain()
    if widget_domain:
        meta["ui"]["domain"] = widget_domain
        meta["openai/widgetDomain"] = widget_domain
    return meta


def _render_recipe_tool_meta(*, app_only: bool = False) -> dict[str, Any]:
    ui_meta: dict[str, Any] = {
        "resourceUri": RECIPE_WIDGET_URI,
    }
    if app_only:
        ui_meta["visibility"] = ["app"]
    return {
        "ui": ui_meta,
        "openai/outputTemplate": RECIPE_WIDGET_URI,
        "openai/toolInvocation/invoking": "Renderizando receta...",
        "openai/toolInvocation/invoked": "Widget listo",
    }


def _query_recipe_data_tool_meta() -> dict[str, Any]:
    return {
        **_render_recipe_tool_meta(),
        "openai/toolInvocation/invoking": "Preparando receta...",
        "openai/toolInvocation/invoked": "Receta lista",
    }


def _build_recipe_widget_html() -> str:
    widget_path = _resolve_recipe_widget_html_path()
    return widget_path.read_text(encoding="utf-8")


def _build_query_recipe_data_tool_result(
    payload: RecipeWidgetPayload,
) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "Mostrando la receta exacta de Mercadona en el widget. "
                    "Evita parafrasear fuera del panel los productos ya "
                    "renderizados."
                ),
            )
        ],
        structuredContent=payload.model_dump(mode="json"),
    )


def _build_render_recipe_tool_result(
    payload: RecipeWidgetPayload,
) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "Mostrando la receta exacta de Mercadona en el panel "
                    "adjunto. Evita repetir o parafrasear fuera del widget "
                    "los productos ya renderizados."
                ),
            )
        ],
        structuredContent=payload.model_dump(mode="json"),
        _meta={
            "recetona/widgetVersion": "recipe-v2",
        },
    )


def _build_search_tool_result(
    results: list[dict[str, str]],
) -> CallToolResult:
    payload = {"results": results}
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False),
            )
        ],
        structuredContent=payload,
    )


def _build_fetch_tool_result(payload: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False),
            )
        ],
        structuredContent=payload,
    )


def create_mcp(
    *,
    host: str = "0.0.0.0",
    port: int = 8788,
    streamable_http_path: str = "/",
    log_level: str = "INFO",
    json_response: bool = False,
    stateless_http: bool = False,
) -> FastMCP:
    mcp_server = FastMCP(
        name=MCP_NAME,
        instructions=(
            "Servidor MCP de la ChatGPT App de RecetONA. "
            "Cuando el usuario pida una receta, usa query_recipe_data para "
            "mostrar directamente el widget final sin parafrasear los "
            "nombres de Mercadona. Usa search y fetch para explorar "
            "productos del catalogo."
        ),
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        log_level=log_level,
        json_response=json_response,
        stateless_http=stateless_http,
    )

    @mcp_server.resource(
        RECIPE_WIDGET_URI,
        name="recipe_widget",
        title="RecetONA Recipe Widget",
        description=(
            "Muestra la receta y la lista de compra exacta de RecetONA."
        ),
        mime_type="text/html;profile=mcp-app",
        meta=_recipe_widget_resource_meta(),
    )
    def recipe_widget() -> str:
        return _build_recipe_widget_html()

    @mcp_server.custom_route("/health", methods=["GET"])
    async def health_route(_request):
        status = _recipe_runtime_status()
        return JSONResponse(
            {
                "ok": True,
                "service": APP_SERVICE_NAME,
                "transport": "streamable-http",
                "mcp_path": streamable_http_path,
                "query_recipe_ready": (
                    status["openai_api_key_present"]
                    and not status["missing_assets"]
                ),
                "recipe_runtime": status,
            }
        )

    @mcp_server.tool(
        name="query_recipe_data",
        description=(
            "Use this when the user asks for a recipe with Mercadona "
            "products and you want to show the final RecetONA widget with "
            "the exact product names, prices, and recipe steps."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
        meta=_query_recipe_data_tool_meta(),
    )
    def query_recipe_data(pregunta: str) -> Any:
        pregunta_normalizada = str(pregunta).strip()
        if not pregunta_normalizada:
            raise ValueError("La pregunta no puede estar vacia.")

        result = _get_service().ask(pregunta_normalizada)
        payload = _build_query_recipe_payload(
            pregunta=pregunta_normalizada,
            result=result,
        )
        return _build_query_recipe_data_tool_result(payload)

    @mcp_server.tool(
        name="render_recipe_widget",
        description=(
            "Use this when you already have the exact structured recipe "
            "payload from query_recipe_data and you want to show the "
            "RecetONA widget. Always call query_recipe_data first and pass "
            "its payload unchanged."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
        meta=_render_recipe_tool_meta(app_only=True),
    )
    def render_recipe_widget(payload: RecipeWidgetPayload) -> Any:
        return _build_render_recipe_tool_result(payload)

    @mcp_server.tool(
        name="search",
        description=(
            "Use this when you need to search the Mercadona catalog for "
            "products related to a query."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def search(query: str, limit: int = 8) -> CallToolResult:
        normalized_query = str(query).strip()
        if not normalized_query:
            return _build_search_tool_result([])

        bounded_limit = max(1, min(int(limit), 25))
        catalog_dataframe = _load_catalog()
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", _normalize(normalized_query))
            if len(token) > 1
        ]
        if not tokens:
            return _build_search_tool_result([])

        scores = np.zeros(len(catalog_dataframe), dtype=np.int16)
        for token in tokens:
            scores += (
                catalog_dataframe["search_text"]
                .str.contains(re.escape(token), regex=True, na=False)
                .to_numpy(dtype=np.int16)
            )

        hit_indices = np.where(scores > 0)[0]
        if len(hit_indices) == 0:
            return _build_search_tool_result([])

        scored_rows = catalog_dataframe.iloc[hit_indices].copy()
        scored_rows["score"] = scores[hit_indices]
        scored_rows = scored_rows.sort_values(
            by=["score", "price_unit"],
            ascending=[False, True],
        ).head(bounded_limit)

        results = []
        for _, row in scored_rows.iterrows():
            product_id = _format_product_id(
                row.get("product_id", row.get("row_idx", ""))
            )
            title = str(
                row.get("product_name", f"Producto {product_id}")
            ).strip()
            results.append(
                {
                    "id": product_id,
                    "title": title,
                    "url": f"recetona://producto/{product_id}",
                }
            )

        return _build_search_tool_result(results)

    @mcp_server.tool(
        name="fetch",
        description=(
            "Use this when you already know a Mercadona product id and need "
            "its full catalog detail."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def fetch(id: str) -> CallToolResult:
        requested_id = str(id).strip()
        if not requested_id:
            raise ValueError("El id no puede estar vacio.")

        catalog_dataframe = _load_catalog()

        product_identifier_column = catalog_dataframe.get(
            "product_id",
            pd.Series(dtype=object),
        ).map(_format_product_id)
        product_mask = product_identifier_column == requested_id
        if product_mask.any():
            row = catalog_dataframe[product_mask].iloc[0]
            return _build_fetch_tool_result(_row_to_fetch_payload(row))

        if "row_idx" in catalog_dataframe.columns:
            row_index_as_text = catalog_dataframe["row_idx"].astype(str)
            row_index_mask = row_index_as_text == requested_id
            if row_index_mask.any():
                row = catalog_dataframe[row_index_mask].iloc[0]
                return _build_fetch_tool_result(_row_to_fetch_payload(row))

        raise ValueError(f"No existe producto con id '{requested_id}'.")

    return mcp_server


def _parse_allowed_origins() -> list[str]:
    raw_value = os.getenv("RECETONA_CORS_ALLOWED_ORIGINS", "").strip()
    if not raw_value:
        return ["*"]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def build_streamable_http_app(
    *,
    json_response: bool = True,
    stateless_http: bool = True,
) -> Any:
    mcp_server = create_mcp(
        json_response=json_response,
        stateless_http=stateless_http,
    )
    application = mcp_server.streamable_http_app()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_allowed_origins(),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"],
    )
    return application


def build_health_payload() -> str:
    return json.dumps(
        {
            "ok": True,
            "service": APP_SERVICE_NAME,
            "recipe_runtime": _recipe_runtime_status(),
        },
        ensure_ascii=False,
    )
