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
from openai import OpenAI
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
RECIPE_WIDGET_URI = "ui://widget/recetona-recipe-v9.html"
MERCADONA_IMAGE_DOMAIN = "https://prod-mercadona.imgix.net"
CATALOG_CACHE_PATH = PROJECT_ROOT / "rag_cache" / "chunks.csv"
EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "rag_cache" / "embeddings.npy"
EXCEL_PATH = PROJECT_ROOT / "mercadona_data.xlsx"

_service: NotebookRagService | None = None
_catalog_dataframe: pd.DataFrame | None = None
_resolved_openai_api_key: str | None = None
_resolved_openai_api_key_source: str | None = None
_recipe_validation_client: OpenAI | None = None
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
    ("curry", re.compile(r"\bcurry\b", re.IGNORECASE)),
    (
        "cayena",
        re.compile(
            r"\b(?:cayena|chile(?: en polvo)?|aji picante|guindilla)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "caldo de pollo",
        re.compile(r"\bcaldo de pollo\b", re.IGNORECASE),
    ),
    (
        "queso crema",
        re.compile(
            r"\b(?:queso crema|queso de untar|queso untable|"
            r"philadelphia|mascarpone)\b",
            re.IGNORECASE,
        ),
    ),
    ("gelatina", re.compile(r"\bgelatina\b", re.IGNORECASE)),
    ("nueces", re.compile(r"\bnue(?:z|ces)\b", re.IGNORECASE)),
    ("pollo", re.compile(r"\bpollo\b", re.IGNORECASE)),
)
FALLBACK_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "azucar": ("azucar", "azúcar"),
    "cacao puro": ("cacao", "puro"),
    "canela": ("canela",),
    "caldo de pollo": ("caldo de pollo", "caldo pollo"),
    "cayena": ("cayena", "guindilla"),
    "curry": ("curry",),
    "gelatina": ("gelatina",),
    "queso crema": (
        "queso crema",
        "queso de untar",
        "queso untable",
        "philadelphia",
        "mascarpone",
    ),
    "chocolate negro": ("chocolate", "negro", "fundir"),
    "gasificante reposteria": ("gasificante", "reposteria"),
    "harina": ("harina",),
    "huevos": ("huevos",),
    "leche": ("leche",),
    "mantequilla": ("mantequilla",),
    "nueces": ("nueces", "nuez"),
    "pollo": ("pollo", "pechuga de pollo", "contramuslos de pollo"),
    "sal": ("sal",),
    "vainilla": ("vainilla",),
}
FLAVORED_GELATIN_TOKENS = {
    "arandano",
    "cereza",
    "cola",
    "fresa",
    "frutos",
    "limon",
    "mandarina",
    "mango",
    "maracuya",
    "pina",
    "proteinas",
    "sabores",
    "sabor",
    "sandia",
    "silvestres",
}
CREAM_CHEESE_FORBIDDEN_TOKENS = {
    "azul",
    "cabra",
    "camembert",
    "cebolla",
    "finas",
    "hierbas",
    "salmon",
    "atun",
}
NON_SHOPPING_INGREDIENTS = {"agua"}


class RecipeIngredientItem(BaseModel):
    ingrediente_objetivo: str | None = None
    producto_mercadona: str
    producto_id: str | None = None
    imagen_url: str | None = None
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
    calorias_totales_kcal: float | None = None
    personas: int | None = None
    ingredientes_inferidos: list[str] = Field(default_factory=list)


class RecipeMissingIngredientsValidation(BaseModel):
    missing_ingredients: list[str] = Field(default_factory=list)


class RecipeIngredientReplacementSuggestion(BaseModel):
    current_product_name: str
    replacement_ingredient: str


class RecipeIngredientReplacementValidation(BaseModel):
    replacements: list[RecipeIngredientReplacementSuggestion] = Field(
        default_factory=list
    )


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


def _get_recipe_validation_model() -> str:
    return (
        os.getenv("RECETONA_RECIPE_VALIDATION_MODEL", "").strip()
        or os.getenv("RECETONA_CHAT_MODEL", "").strip()
        or "gpt-5.4-nano"
    )


def _get_recipe_validation_reasoning() -> dict[str, str] | None:
    effort = (
        os.getenv("RECETONA_RECIPE_VALIDATION_REASONING_EFFORT", "")
        .strip()
        .lower()
        or os.getenv("RECETONA_REASONING_EFFORT", "").strip().lower()
        or "none"
    )
    if not effort:
        return None
    return {"effort": effort}


def _get_recipe_validation_client() -> OpenAI:
    global _recipe_validation_client

    if _recipe_validation_client is not None:
        return _recipe_validation_client

    api_key = _resolve_openai_api_key(raise_on_error=True)
    _recipe_validation_client = OpenAI(api_key=api_key)
    return _recipe_validation_client


def _extract_missing_ingredients_from_validation_response(
    response: Any,
) -> list[str]:
    missing_ingredients: list[str] = []
    seen: set[str] = set()

    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue

        for content_item in getattr(output, "content", []):
            if getattr(content_item, "type", None) == "refusal":
                return []

            parsed_payload = getattr(content_item, "parsed", None)
            if parsed_payload is None:
                continue

            for item in getattr(parsed_payload, "missing_ingredients", []):
                normalized_item = _canonicalize_catalog_ingredient_name(item)
                if not normalized_item or normalized_item in seen:
                    continue
                seen.add(normalized_item)
                missing_ingredients.append(normalized_item)

    return missing_ingredients


def _extract_replacement_suggestions_from_validation_response(
    response: Any,
) -> list[RecipeIngredientReplacementSuggestion]:
    suggestions: list[RecipeIngredientReplacementSuggestion] = []
    seen: set[tuple[str, str]] = set()

    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue

        for content_item in getattr(output, "content", []):
            if getattr(content_item, "type", None) == "refusal":
                return []

            parsed_payload = getattr(content_item, "parsed", None)
            if parsed_payload is None:
                continue

            for item in getattr(parsed_payload, "replacements", []):
                product_name = str(
                    getattr(item, "current_product_name", "") or ""
                ).strip()
                replacement_ingredient = _canonicalize_catalog_ingredient_name(
                    getattr(item, "replacement_ingredient", "")
                )
                if not product_name or not replacement_ingredient:
                    continue
                key = (
                    _normalize_matching_text(product_name),
                    replacement_ingredient,
                )
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(
                    RecipeIngredientReplacementSuggestion(
                        current_product_name=product_name,
                        replacement_ingredient=replacement_ingredient,
                    )
                )

    return suggestions


def _normalize_existing_ingredient_tokens(
    ingredients: list[dict[str, Any]],
) -> set[str]:
    normalized_tokens: set[str] = set()

    for ingredient in ingredients:
        for raw_value in (
            ingredient.get("ingrediente_objetivo"),
            ingredient.get("producto_mercadona"),
        ):
            normalized_value = _normalize_matching_text(raw_value)
            if not normalized_value:
                continue
            normalized_tokens.add(normalized_value)
            normalized_tokens.update(
                token
                for token in re.findall(r"[a-z0-9]+", normalized_value)
                if len(token) >= 3
            )

    return normalized_tokens


def _normalized_text_tokens(value: Any) -> set[str]:
    normalized_value = _normalize_matching_text(value)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized_value)
        if len(token) >= 3
    }


def _canonicalize_catalog_ingredient_name(value: Any) -> str:
    normalized_value = _normalize_matching_text(value)
    if not normalized_value:
        return ""

    for ingredient_name, pattern in FALLBACK_INGREDIENT_PATTERNS:
        if pattern.search(normalized_value):
            return ingredient_name

    return normalized_value


def _ingredient_appears_only_in_optional_recipe_context(
    *,
    ingredient: str,
    recipe_text: str,
) -> bool:
    normalized_ingredient = _canonicalize_catalog_ingredient_name(ingredient)
    normalized_recipe_text = _normalize_matching_text(recipe_text)
    if not normalized_ingredient or not normalized_recipe_text:
        return False

    ingredient_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized_ingredient)
        if len(token) >= 3
    ]
    if not ingredient_tokens:
        return False

    recipe_sentences = re.split(r"(?<=[.!?])\s+|\n+", normalized_recipe_text)
    matched_sentences = [
        sentence
        for sentence in recipe_sentences
        if sentence and any(token in sentence for token in ingredient_tokens)
    ]
    if not matched_sentences:
        return False

    optional_sentence_pattern = re.compile(
        r"\b(?:sirve|servir|acompa(?:n|ñ)ad[oa]|guarnicion|opcional|"
        r"si se desea|al gusto)\b"
    )

    if normalized_ingredient == "agua":
        return all(
            "si es necesario" in sentence for sentence in matched_sentences
        )

    return all(
        optional_sentence_pattern.search(sentence)
        for sentence in matched_sentences
    )


def _selected_ingredient_covers_target(
    *,
    target_ingredient: str,
    selected_ingredient: dict[str, Any],
) -> bool:
    normalized_target = _canonicalize_catalog_ingredient_name(
        target_ingredient
    )
    if not normalized_target:
        return False

    target_tokens = _normalized_text_tokens(normalized_target)
    for raw_value in (
        selected_ingredient.get("ingrediente_objetivo"),
        selected_ingredient.get("producto_mercadona"),
    ):
        normalized_value = _normalize_matching_text(raw_value)
        if not normalized_value:
            continue
        if (
            normalized_target == normalized_value
            or normalized_target in normalized_value
            or normalized_value in normalized_target
        ):
            return True
        if target_tokens and target_tokens <= _normalized_text_tokens(
            normalized_value
        ):
            return True

    catalog_row = _find_catalog_row_for_ingredient_item(selected_ingredient)
    if catalog_row is not None and normalized_target in (
        FALLBACK_SEARCH_TERMS.keys() | {"gelatina", "nueces", "pollo"}
    ):
        return _row_is_compatible_for_fallback_ingredient(
            ingredient=normalized_target,
            row=catalog_row,
        )

    return False


def _text_explicitly_mentions_ingredient(
    *,
    ingredient: str,
    text: str,
) -> bool:
    normalized_ingredient = _canonicalize_catalog_ingredient_name(ingredient)
    normalized_text = _normalize_matching_text(text)
    if not normalized_ingredient or not normalized_text:
        return False

    ingredient_tokens = _normalized_text_tokens(normalized_ingredient)
    if not ingredient_tokens:
        return False

    text_tokens = _normalized_text_tokens(normalized_text)
    return bool(ingredient_tokens <= text_tokens)


def _collect_missing_inferred_ingredients(
    *,
    pregunta: str,
    recipe_text: str,
    ingredients: list[dict[str, Any]],
    inferred_ingredients: list[str],
) -> list[str]:
    missing_ingredients: list[str] = []
    seen: set[str] = set()

    for inferred_ingredient in inferred_ingredients:
        normalized_target = _canonicalize_catalog_ingredient_name(
            inferred_ingredient
        )
        if not normalized_target or normalized_target in seen:
            continue
        if normalized_target in NON_SHOPPING_INGREDIENTS:
            continue
        if not _text_explicitly_mentions_ingredient(
            ingredient=normalized_target,
            text=pregunta,
        ) and _ingredient_appears_only_in_optional_recipe_context(
            ingredient=normalized_target,
            recipe_text=recipe_text,
        ):
            continue
        if any(
            _selected_ingredient_covers_target(
                target_ingredient=normalized_target,
                selected_ingredient=ingredient,
            )
            for ingredient in ingredients
        ):
            continue
        seen.add(normalized_target)
        missing_ingredients.append(normalized_target)

    return missing_ingredients


def _build_rule_based_replacement_suggestions(
    ingredients: list[dict[str, Any]],
) -> list[RecipeIngredientReplacementSuggestion]:
    supported_ingredients = {
        "cayena",
        "curry",
        "gelatina",
        "pollo",
        "queso crema",
        "tomate triturado",
    }
    suggestions: list[RecipeIngredientReplacementSuggestion] = []
    seen: set[tuple[str, str]] = set()

    for ingredient in ingredients:
        target_ingredient = _canonicalize_catalog_ingredient_name(
            ingredient.get("ingrediente_objetivo")
        )
        if target_ingredient not in supported_ingredients:
            continue

        product_name = str(ingredient.get("producto_mercadona") or "").strip()
        if not product_name:
            continue

        catalog_row = _find_catalog_row_for_ingredient_item(ingredient)
        if catalog_row is None:
            continue

        if _row_is_compatible_for_fallback_ingredient(
            ingredient=target_ingredient,
            row=catalog_row,
        ):
            continue

        dedupe_key = (
            _normalize_matching_text(product_name),
            target_ingredient,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        suggestions.append(
            RecipeIngredientReplacementSuggestion(
                current_product_name=product_name,
                replacement_ingredient=target_ingredient,
            )
        )

    return suggestions


def _validate_recipe_ingredients_with_agent(
    *,
    pregunta: str,
    recipe_text: str,
    ingredients: list[dict[str, Any]],
) -> list[str]:
    if not recipe_text.strip():
        return []

    current_lines: list[str] = []
    for ingredient in ingredients:
        ingredient_target = str(
            ingredient.get("ingrediente_objetivo") or ""
        ).strip()
        product_name = str(ingredient.get("producto_mercadona") or "").strip()
        if not ingredient_target and not product_name:
            continue
        current_lines.append(
            f"- ingrediente_objetivo={ingredient_target or 'N/D'}"
            f" | producto_mercadona={product_name or 'N/D'}"
        )

    current_list = "\n".join(current_lines) or "- lista vacia"
    user_prompt = (
        "Consulta del usuario:\n"
        f"{pregunta}\n\n"
        "Receta generada:\n"
        f"{recipe_text}\n\n"
        "Lista actual de ingredientes/productos de Mercadona:\n"
        f"{current_list}\n\n"
        "Detecta ingredientes nucleares que la receta usa de verdad pero que "
        "no estan presentes en la lista actual. Devuelve nombres cortos, "
        "normalizados y buscables en catalogo como 'pollo', 'cebolla', "
        "'nata para montar' o 'gelatina neutra'. No devuelvas cantidades, "
        "marcas ni frases completas."
    )

    try:
        request_kwargs: dict[str, Any] = {
            "model": _get_recipe_validation_model(),
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Eres un validador de recetas para RecetONA. "
                                "Compara los pasos de la receta con la lista "
                                "de productos ya elegidos. Devuelve solo los "
                                "ingredientes esenciales que faltan realmente "
                                "en la lista. Excluye guarniciones opcionales, "
                                "sugerencias de servicio, ingredientes "
                                "implícitos no mencionados y duplicados."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                },
            ],
            "text_format": RecipeMissingIngredientsValidation,
            "temperature": 0,
            "max_output_tokens": 200,
        }
        reasoning = _get_recipe_validation_reasoning()
        if reasoning:
            request_kwargs["reasoning"] = reasoning
        response = _get_recipe_validation_client().responses.parse(
            **request_kwargs
        )
    except Exception:
        return []

    missing_ingredients = (
        _extract_missing_ingredients_from_validation_response(response)
    )
    if not missing_ingredients:
        return []

    existing_tokens = _normalize_existing_ingredient_tokens(ingredients)
    filtered_missing_ingredients: list[str] = []
    seen: set[str] = set()

    for item in missing_ingredients:
        if item in NON_SHOPPING_INGREDIENTS:
            continue
        if item in seen or item in existing_tokens:
            continue
        item_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", item)
            if len(token) >= 3
        }
        if item_tokens and item_tokens <= existing_tokens:
            continue
        if _ingredient_appears_only_in_optional_recipe_context(
            ingredient=item,
            recipe_text=recipe_text,
        ):
            continue
        seen.add(item)
        filtered_missing_ingredients.append(item)

    return filtered_missing_ingredients


def _validate_recipe_substitutions_with_agent(
    *,
    pregunta: str,
    recipe_text: str,
    ingredients: list[dict[str, Any]],
) -> list[RecipeIngredientReplacementSuggestion]:
    if not recipe_text.strip() or not ingredients:
        return []

    current_lines: list[str] = []
    for ingredient in ingredients:
        ingredient_target = str(
            ingredient.get("ingrediente_objetivo") or ""
        ).strip()
        product_name = str(ingredient.get("producto_mercadona") or "").strip()
        if not product_name:
            continue
        current_lines.append(
            f"- ingrediente_objetivo={ingredient_target or 'N/D'}"
            f" | producto_mercadona={product_name}"
        )

    current_list = "\n".join(current_lines)
    user_prompt = (
        "Consulta del usuario:\n"
        f"{pregunta}\n\n"
        "Receta generada:\n"
        f"{recipe_text}\n\n"
        "Lista actual de ingredientes/productos de Mercadona:\n"
        f"{current_list}\n\n"
        "Detecta solo productos claramente mal resueltos o implausibles para "
        "la receta. Devuelve el nombre literal exacto del producto actual y "
        "una consulta corta del ingrediente correcto para buscarlo en el "
        "catalogo, por ejemplo 'curry' o 'cayena'. No marques equivalencias "
        "razonables ni guarniciones opcionales."
    )

    try:
        request_kwargs: dict[str, Any] = {
            "model": _get_recipe_validation_model(),
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Eres un validador de plausibilidad para "
                                "RecetONA. Compara la receta con la lista de "
                                "productos ya elegidos. Devuelve solo los "
                                "casos obvios en los que el producto de "
                                "Mercadona no corresponde al ingrediente "
                                "realmente usado en la receta. No marques "
                                "aceite de oliva para aceite, ni variantes "
                                "cercanas razonables. Si un producto es "
                                "claramente incorrecto, devuelve el nombre "
                                "literal exacto del producto actual y un "
                                "ingrediente corto para buscar el reemplazo."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                },
            ],
            "text_format": RecipeIngredientReplacementValidation,
            "temperature": 0,
            "max_output_tokens": 250,
        }
        reasoning = _get_recipe_validation_reasoning()
        if reasoning:
            request_kwargs["reasoning"] = reasoning
        response = _get_recipe_validation_client().responses.parse(
            **request_kwargs
        )
    except Exception:
        return []

    suggestions = _extract_replacement_suggestions_from_validation_response(
        response
    )
    rule_based_suggestions = _build_rule_based_replacement_suggestions(
        ingredients
    )
    if not suggestions and not rule_based_suggestions:
        return []

    existing_products = {
        _normalize_matching_text(ingredient.get("producto_mercadona"))
        for ingredient in ingredients
        if _normalize_matching_text(ingredient.get("producto_mercadona"))
    }
    filtered_suggestions: list[RecipeIngredientReplacementSuggestion] = []
    seen_keys: set[tuple[str, str]] = set()

    for suggestion in list(rule_based_suggestions) + list(suggestions):
        normalized_current_product = _normalize_matching_text(
            suggestion.current_product_name
        )
        if normalized_current_product not in existing_products:
            continue
        if _ingredient_appears_only_in_optional_recipe_context(
            ingredient=suggestion.replacement_ingredient,
            recipe_text=recipe_text,
        ):
            continue
        dedupe_key = (
            normalized_current_product,
            suggestion.replacement_ingredient,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        filtered_suggestions.append(suggestion)

    return filtered_suggestions


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


def _quantity_to_base(
    value: Any,
    unit: Any,
) -> tuple[str | None, float | None]:
    numeric_value = _safe_float_or_none(value)
    unit_text = _normalize_matching_text(unit)
    if numeric_value is None:
        return None, None

    if unit_text == "kg":
        return "mass_kg", numeric_value
    if unit_text in {"g", "gr", "gramo", "gramos"}:
        return "mass_kg", numeric_value / 1000.0
    if unit_text == "l":
        return "vol_l", numeric_value
    if unit_text == "ml":
        return "vol_l", numeric_value / 1000.0
    if unit_text in {"ud", "u", "unidad", "unidades"}:
        return "unit_ud", numeric_value

    return None, None


def _kcal_for_base_quantity(
    *,
    quantity_kind: str | None,
    quantity_value: float | None,
    nutrition_kcal_100: float | None,
) -> float | None:
    if (
        quantity_kind not in {"mass_kg", "vol_l"}
        or quantity_value is None
        or nutrition_kcal_100 is None
    ):
        return None

    return nutrition_kcal_100 * quantity_value * 10.0


def _extract_first_image_url(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            image_url = _extract_first_image_url(item)
            if image_url:
                return image_url
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    if text.startswith("["):
        try:
            parsed_value = json.loads(text)
        except Exception:
            parsed_value = None
        if parsed_value is not None:
            return _extract_first_image_url(parsed_value)

    for separator in (" | ", "|", ","):
        if separator in text:
            for chunk in text.split(separator):
                image_url = _extract_first_image_url(chunk)
                if image_url:
                    return image_url
            return None

    if text.startswith("http://") or text.startswith("https://"):
        return text

    return None


def _candidate_recipe_aliases(ingredient: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    for raw_value in (
        ingredient.get("producto_mercadona"),
        ingredient.get("ingrediente_objetivo"),
    ):
        alias = str(raw_value or "").strip()
        normalized_alias = _normalize_matching_text(alias)
        if not normalized_alias or normalized_alias in seen:
            continue
        seen.add(normalized_alias)
        aliases.append(alias)

    return sorted(aliases, key=len, reverse=True)


def _extract_recipe_quantity_for_ingredient(
    recipe_text: str,
    ingredient: dict[str, Any],
) -> tuple[str | None, float | None]:
    if not recipe_text.strip():
        return None, None

    totals: dict[str, float] = {
        "mass_kg": 0.0,
        "vol_l": 0.0,
        "unit_ud": 0.0,
    }
    matched_spans: list[tuple[int, int]] = []

    for alias in _candidate_recipe_aliases(ingredient):
        alias_pattern = re.escape(alias)
        patterns = (
            re.compile(
                rf"(?P<value>\d+(?:[.,]\d+)?)\s*"
                rf"(?P<unit>kg|g|gr|gramos|l|ml|ud|u|unidad|unidades)"
                rf"\s+de\s+{alias_pattern}\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"(?P<value>\d+(?:[.,]\d+)?)\s*"
                rf"(?P<unit>ud|u|unidad|unidades)?\s*"
                rf"{alias_pattern}\b",
                re.IGNORECASE,
            ),
        )

        for pattern in patterns:
            for match in pattern.finditer(recipe_text):
                span = match.span()
                if any(
                    span[0] < existing_end and existing_start < span[1]
                    for existing_start, existing_end in matched_spans
                ):
                    continue
                matched_spans.append(span)

                unit = match.groupdict().get("unit") or "ud"
                kind, value = _quantity_to_base(
                    match.group("value").replace(",", "."),
                    unit,
                )
                if kind is None or value is None:
                    continue
                totals[kind] += value

    for kind in ("mass_kg", "vol_l", "unit_ud"):
        if totals[kind] > 0:
            return kind, totals[kind]

    return None, None


def _find_catalog_row_for_ingredient_item(
    ingredient: dict[str, Any],
) -> pd.Series | None:
    catalog_dataframe = _load_catalog()
    product_id = str(ingredient.get("producto_id") or "").strip()
    product_name = str(ingredient.get("producto_mercadona") or "").strip()

    matches = pd.DataFrame()
    if product_id and "product_id" in catalog_dataframe.columns:
        matches = catalog_dataframe[
            catalog_dataframe["product_id"].apply(
                lambda value: (
                    _format_product_id(value) == product_id
                    if value is not None and not pd.isna(value)
                    else False
                )
            )
        ]

    if matches.empty and product_name:
        matches = catalog_dataframe[
            catalog_dataframe["product_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(product_name)
        ]

    if matches.empty:
        return None

    package_value = _safe_float_or_none(ingredient.get("tamano_envase_valor"))
    package_unit = _normalize_matching_text(
        ingredient.get("tamano_envase_unidad")
    )
    if package_value is not None and package_unit:
        sized_matches = matches[
            matches["unit_size"].apply(
                lambda value: _safe_float_or_none(value) == package_value
            )
            & matches["size_format"]
            .fillna("")
            .astype(str)
            .map(_normalize_matching_text)
            .eq(package_unit)
        ]
        if not sized_matches.empty:
            matches = sized_matches

    return matches.iloc[0]


def _resolve_catalog_image_url(catalog_row: pd.Series | None) -> str | None:
    if catalog_row is None:
        return None

    for field_name in ("thumbnail_url", "photo_urls"):
        image_url = _extract_first_image_url(catalog_row.get(field_name))
        if image_url:
            return image_url

    return None


def _enrich_ingredient_with_catalog_metadata(
    ingredient: dict[str, Any],
) -> dict[str, Any]:
    enriched_ingredient = dict(ingredient)
    if enriched_ingredient.get("imagen_url"):
        return enriched_ingredient

    catalog_row = _find_catalog_row_for_ingredient_item(enriched_ingredient)
    image_url = _resolve_catalog_image_url(catalog_row)
    if image_url:
        enriched_ingredient["imagen_url"] = image_url

    return enriched_ingredient


def _estimate_ingredient_consumed_kcal(
    ingredient: dict[str, Any],
    recipe_text: str,
) -> float | None:
    catalog_row = _find_catalog_row_for_ingredient_item(ingredient)
    if catalog_row is None:
        return None

    nutrition_kcal_100 = _safe_float_or_none(
        catalog_row.get("nutrition_kcal_100")
    )
    if nutrition_kcal_100 is None:
        return None

    explicit_kind, explicit_value = _extract_recipe_quantity_for_ingredient(
        recipe_text,
        ingredient,
    )

    package_kind, package_value = _quantity_to_base(
        ingredient.get("tamano_envase_valor"),
        ingredient.get("tamano_envase_unidad"),
    )
    package_kcal = _kcal_for_base_quantity(
        quantity_kind=package_kind,
        quantity_value=package_value,
        nutrition_kcal_100=nutrition_kcal_100,
    )

    explicit_kcal = _kcal_for_base_quantity(
        quantity_kind=explicit_kind,
        quantity_value=explicit_value,
        nutrition_kcal_100=nutrition_kcal_100,
    )
    if explicit_kcal is not None:
        return explicit_kcal

    purchase_cost = _safe_float_or_none(ingredient.get("coste_compra_eur"))
    consumed_cost = _safe_float_or_none(ingredient.get("coste_consumido_eur"))
    if (
        package_kcal is not None
        and purchase_cost is not None
        and purchase_cost > 0
        and consumed_cost is not None
        and consumed_cost >= 0
    ):
        return package_kcal * (consumed_cost / purchase_cost)

    quantity_kind, quantity_value = _quantity_to_base(
        ingredient.get("cantidad_receta_valor"),
        ingredient.get("cantidad_receta_unidad"),
    )
    direct_kcal = _kcal_for_base_quantity(
        quantity_kind=quantity_kind,
        quantity_value=quantity_value,
        nutrition_kcal_100=nutrition_kcal_100,
    )
    if direct_kcal is not None:
        return direct_kcal

    return None


def _estimate_total_recipe_kcal(
    ingredients: list[dict[str, Any]],
    recipe_text: str,
) -> float | None:
    total_kcal = 0.0
    has_any_value = False

    for ingredient in ingredients:
        ingredient_kcal = _estimate_ingredient_consumed_kcal(
            ingredient,
            recipe_text,
        )
        if ingredient_kcal is None:
            continue
        total_kcal += ingredient_kcal
        has_any_value = True

    if not has_any_value:
        return None

    return round(total_kcal, 1)


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


def _merge_inferred_ingredients(
    current_ingredients: list[str],
    new_ingredients: list[str],
) -> list[str]:
    merged_ingredients: list[str] = []
    seen: set[str] = set()

    for item in list(current_ingredients) + list(new_ingredients):
        normalized_item = _normalize_matching_text(item)
        if not normalized_item or normalized_item in seen:
            continue
        seen.add(normalized_item)
        merged_ingredients.append(normalized_item)

    return merged_ingredients


def _replace_implausible_ingredients(
    *,
    ingredients: list[dict[str, Any]],
    replacements: list[RecipeIngredientReplacementSuggestion],
    pregunta: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    repaired_ingredients = list(ingredients)
    applied_replacements: list[str] = []

    for replacement in replacements:
        normalized_product_name = _normalize_matching_text(
            replacement.current_product_name
        )
        replacement_index = next(
            (
                index
                for index, ingredient in enumerate(repaired_ingredients)
                if _normalize_matching_text(
                    ingredient.get("producto_mercadona")
                )
                == normalized_product_name
            ),
            None,
        )
        if replacement_index is None:
            continue

        selected_products = {
            str(ingredient.get("producto_mercadona") or "").strip()
            for index, ingredient in enumerate(repaired_ingredients)
            if index != replacement_index
            and str(ingredient.get("producto_mercadona") or "").strip()
        }
        replacement_items = _build_catalog_ingredient_items(
            ingredients=[replacement.replacement_ingredient],
            pregunta=pregunta,
            selected_products=selected_products,
            note=("Reemplazado tras validar plausibilidad frente a la receta"),
        )
        if not replacement_items:
            continue

        repaired_ingredients[replacement_index] = replacement_items[
            0
        ].model_dump(mode="json")
        applied_replacements.append(replacement.replacement_ingredient)

    return repaired_ingredients, applied_replacements


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
    recipe_text = str(result.get("block_3", "")).strip()

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

    validation_missing_ingredients = _validate_recipe_ingredients_with_agent(
        pregunta=pregunta,
        recipe_text=recipe_text,
        ingredients=ingredients,
    )
    if validation_missing_ingredients:
        existing_products = {
            str(ingredient.get("producto_mercadona") or "").strip()
            for ingredient in ingredients
            if str(ingredient.get("producto_mercadona") or "").strip()
        }
        added_ingredients = _build_catalog_ingredient_items(
            ingredients=validation_missing_ingredients,
            pregunta=pregunta,
            selected_products=existing_products,
            note=(
                "Anadido tras validar la receta frente a la lista final "
                "de ingredientes"
            ),
        )
        if added_ingredients:
            ingredients.extend(
                ingredient.model_dump(mode="json")
                for ingredient in added_ingredients
            )
            inferred_ingredients = _merge_inferred_ingredients(
                list(inferred_ingredients),
                validation_missing_ingredients,
            )
            current_purchase_total = (
                _safe_float_or_none(cost_summary.get("total_purchase_eur"))
                or 0.0
            )
            added_purchase_total = sum(
                ingredient.coste_compra_eur or 0.0
                for ingredient in added_ingredients
            )
            cost_summary["total_purchase_eur"] = round(
                current_purchase_total + added_purchase_total,
                2,
            )

    replacement_suggestions = _validate_recipe_substitutions_with_agent(
        pregunta=pregunta,
        recipe_text=recipe_text,
        ingredients=ingredients,
    )
    applied_replacements: list[str] = []
    if replacement_suggestions:
        ingredients, applied_replacements = _replace_implausible_ingredients(
            ingredients=ingredients,
            replacements=replacement_suggestions,
            pregunta=pregunta,
        )
        if applied_replacements:
            inferred_ingredients = _merge_inferred_ingredients(
                list(inferred_ingredients),
                applied_replacements,
            )

    inferred_missing_ingredients = _collect_missing_inferred_ingredients(
        pregunta=pregunta,
        recipe_text=recipe_text,
        ingredients=ingredients,
        inferred_ingredients=list(inferred_ingredients),
    )
    if inferred_missing_ingredients:
        existing_products = {
            str(ingredient.get("producto_mercadona") or "").strip()
            for ingredient in ingredients
            if str(ingredient.get("producto_mercadona") or "").strip()
        }
        added_inferred_ingredients = _build_catalog_ingredient_items(
            ingredients=inferred_missing_ingredients,
            pregunta=pregunta,
            selected_products=existing_products,
            note=(
                "Anadido desde ingredientes inferidos tras validar "
                "consistencia final de la receta"
            ),
        )
        if added_inferred_ingredients:
            ingredients.extend(
                ingredient.model_dump(mode="json")
                for ingredient in added_inferred_ingredients
            )
            inferred_ingredients = _merge_inferred_ingredients(
                list(inferred_ingredients),
                inferred_missing_ingredients,
            )

    literal_recipe_ingredients = [
        ingredient_name
        for ingredient_name in _extract_fallback_ingredients(
            pregunta=pregunta,
            result=result,
        )
        if any(
            token in ingredient_name
            for token in ("caldo de pollo", "curry", "gelatina", "pollo")
        )
    ]
    missing_literal_recipe_ingredients = [
        ingredient_name
        for ingredient_name in literal_recipe_ingredients
        if not any(
            _selected_ingredient_covers_target(
                target_ingredient=ingredient_name,
                selected_ingredient=ingredient,
            )
            for ingredient in ingredients
        )
    ]
    if missing_literal_recipe_ingredients:
        existing_products = {
            str(ingredient.get("producto_mercadona") or "").strip()
            for ingredient in ingredients
            if str(ingredient.get("producto_mercadona") or "").strip()
        }
        added_literal_recipe_ingredients = _build_catalog_ingredient_items(
            ingredients=missing_literal_recipe_ingredients,
            pregunta=pregunta,
            selected_products=existing_products,
            note=(
                "Anadido desde menciones literales detectadas en la receta "
                "final"
            ),
        )
        if added_literal_recipe_ingredients:
            ingredients.extend(
                ingredient.model_dump(mode="json")
                for ingredient in added_literal_recipe_ingredients
            )
            inferred_ingredients = _merge_inferred_ingredients(
                list(inferred_ingredients),
                missing_literal_recipe_ingredients,
            )

    ingredients = [
        _enrich_ingredient_with_catalog_metadata(ingredient)
        for ingredient in ingredients
    ]
    if ingredients:
        cost_summary["total_purchase_eur"] = round(
            sum(
                _safe_float_or_none(ingredient.get("coste_compra_eur")) or 0.0
                for ingredient in ingredients
            ),
            2,
        )

    ingredientes_texto_literal = str(result.get("block_1", "")).strip()
    if ingredients and (
        not ingredientes_texto_literal
        or "No se encontraron ingredientes/productos"
        in ingredientes_texto_literal
        or validation_missing_ingredients
        or applied_replacements
    ):
        ingredientes_texto_literal = _build_ingredient_lines_text(ingredients)

    respuesta_literal_mcp = str(result.get("answer", "")).strip()
    if ingredients and (
        "No se encontraron ingredientes/productos" in respuesta_literal_mcp
        or validation_missing_ingredients
        or applied_replacements
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
        receta_y_pasos_texto_literal=recipe_text,
        respuesta_literal_mcp=respuesta_literal_mcp,
        coste_total_compra_eur=_safe_float_or_none(
            cost_summary.get("total_purchase_eur")
        ),
        coste_total_consumido_eur=_safe_float_or_none(
            cost_summary.get("total_escandallo_eur")
        ),
        calorias_totales_kcal=_estimate_total_recipe_kcal(
            ingredients,
            recipe_text,
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
    if ingredient == "cayena":
        return (
            "cayena" in product_name or "guindilla" in product_name
        ) and "especias" in taxonomy
    if ingredient == "caldo de pollo":
        return "caldo" in product_name and (
            "pollo" in product_name or "pollo" in taxonomy
        )
    if ingredient == "curry":
        return "curry" in product_name and (
            "especias" in taxonomy or "salsa" in taxonomy
        )
    if ingredient == "queso crema":
        if any(
            token in product_name for token in CREAM_CHEESE_FORBIDDEN_TOKENS
        ):
            return False
        return (
            "philadelphia" in product_name
            or "mascarpone" in product_name
            or (
                "queso" in product_name
                and ("untar" in product_name or "untable" in taxonomy)
            )
        )
    if ingredient == "gelatina":
        return "gelatina" in product_name and not (
            any(token in product_name for token in FLAVORED_GELATIN_TOKENS)
            or "postres" in taxonomy
            or "sabor" in product_name
        )
    if ingredient == "nueces":
        return "nuez" in product_name or "nueces" in product_name
    if "pollo" in ingredient:
        return "pollo" in product_name and (
            "pollo" in taxonomy
            or "aves" in taxonomy
            or "ave" in taxonomy
            or "carne" in taxonomy
        )

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

    return _build_catalog_ingredient_items(
        ingredients=fallback_ingredients,
        pregunta=pregunta,
        selected_products=None,
        note="Fallback desde catalogo por ausencia de cost_plan",
    )


def _build_catalog_ingredient_items(
    *,
    ingredients: list[str],
    pregunta: str,
    selected_products: set[str] | None,
    note: str,
) -> list[RecipeIngredientItem]:
    catalog_dataframe = _load_catalog()
    selected_items: list[RecipeIngredientItem] = []
    selected_product_names = {
        _normalize_matching_text(product_name)
        for product_name in (selected_products or set())
        if _normalize_matching_text(product_name)
    }

    for ingredient in ingredients:
        normalized_ingredient = _canonicalize_catalog_ingredient_name(
            ingredient
        )
        if not normalized_ingredient:
            continue
        if normalized_ingredient in NON_SHOPPING_INGREDIENTS:
            continue

        scored_candidates: list[tuple[int, float, pd.Series]] = []
        for _, row in catalog_dataframe.iterrows():
            if _candidate_is_incompatible_for_ingredient(
                normalized_ingredient,
                row,
                recipe_query=pregunta,
            ):
                continue
            if not _row_is_compatible_for_fallback_ingredient(
                ingredient=normalized_ingredient,
                row=row,
            ):
                continue
            score = _score_catalog_row_for_ingredient(
                ingredient=normalized_ingredient,
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
                if _normalize_matching_text(row.get("product_name"))
                not in selected_product_names
            ),
            scored_candidates[0][2],
        )

        product_name = _normalize_matching_text(best_row.get("product_name"))
        if product_name:
            selected_product_names.add(product_name)

        selected_item = _build_fallback_ingredient_item(
            ingredient=normalized_ingredient,
            row=best_row,
        )
        selected_item.notas = note
        selected_items.append(selected_item)

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
    resource_domains = [MERCADONA_IMAGE_DOMAIN]
    meta = {
        "ui": {
            "prefersBorder": True,
            "csp": {
                "connectDomains": [],
                "resourceDomains": resource_domains,
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
            "resource_domains": resource_domains,
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
                text="Receta mostrada en el widget.",
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
            "recetona/widgetVersion": "recipe-v9",
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
