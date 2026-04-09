#!/usr/bin/env python3.12
import argparse
import json
import logging
import os
import re
import sys
import threading
import traceback
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "mercadona_rag_notebook.ipynb"
ENV_PATH = BASE_DIR / ".env"
RUNTIME_BASE_DIR_ENV = "RECETONA_RUNTIME_BASE_DIR"
RUNTIME_CACHE_DIR_ENV = "RECETONA_RAG_CACHE_DIR"
RAG_CACHE_S3_BUCKET_ENV = "RECETONA_RAG_CACHE_S3_BUCKET"
RAG_CACHE_S3_PREFIX_ENV = "RECETONA_RAG_CACHE_S3_PREFIX"
REQUIRED_RAG_CACHE_FILES = ("chunks.csv", "embeddings.npy")
RAW_NUT_INGREDIENT_TOKENS = {
    "almendra",
    "almendras",
    "anacardo",
    "anacardos",
    "avellana",
    "avellanas",
    "cacahuete",
    "cacahuetes",
    "nuez",
    "nueces",
    "pistacho",
    "pistachos",
}
CARROT_NAME_TOKENS = {"zanahoria", "zanahorias"}
GARLIC_NAME_TOKENS = {"ajo", "ajos"}
GARLIC_PREPARED_FOOD_TOKENS = {
    "alioli",
    "allioli",
    "alino",
    "alinados",
    "mousse",
    "pan",
    "picatostes",
    "rallado",
    "sal",
    "salsa",
    "sazonador",
    "tostado",
}
BEEF_STEW_FORBIDDEN_TOKENS = {
    "albondiga",
    "albondigas",
    "burger",
    "burgers",
    "hamburguesa",
    "hamburguesas",
    "picada",
}
BEEF_NAME_TOKENS = {
    "anojo",
    "buey",
    "ternera",
    "vaca",
    "vacuno",
}
PREPARED_FOOD_PATTERN = re.compile(
    r"\b("
    r"barquillo|barritas?|batido|bebida|bifidus|bizcocho|"
    r"bolleria|bombon|brownie|cereal(?:es)?|cookie|"
    r"croissant|dona|donut|galletas?|granola|helado|"
    r"lacteo|muesli|muffin|napolitana|pastel|pasteleria|"
    r"postre|tarta|trenza|untable|yogur(?:t)?"
    r")\b"
)
CARROT_INGREDIENT_PATTERN = re.compile(r"\bzanahori(?:a|as)\b")
GARLIC_INGREDIENT_PATTERN = re.compile(r"\baj(?:o|os)\b")
CRUSHED_TOMATO_INGREDIENT_PATTERN = re.compile(r"\btomate triturad[oa]?\b")
BEEF_BROTH_INGREDIENT_PATTERN = re.compile(
    r"\bcaldo de (?:carne|res|vacuno|ternera)\b"
)
GELATIN_INGREDIENT_PATTERN = re.compile(r"\bgelatina\b")
CREAM_CHEESE_INGREDIENT_PATTERN = re.compile(
    r"\b(?:queso crema|queso de untar|queso untable|philadelphia|"
    r"mascarpone)\b"
)
PAPRIKA_INGREDIENT_PATTERN = re.compile(r"\bpimenton\b")
BEEF_INGREDIENT_PATTERN = re.compile(
    r"\b(?:carne de (?:res|vacuno|ternera)|vacuno|ternera)\b"
)
STEW_RECIPE_PATTERN = re.compile(
    r"\b(?:estofad\w*|guis\w*|ragu\w*|cocid\w*)\b"
)
NON_RECIPE_TAXONOMY_PATTERN = re.compile(
    r"\b("
    r"botiquin|colonia|cuidado|farmacia|fitoterapia|gato|"
    r"higiene|hogar|limpieza|maquillaje|mascotas?|"
    r"parafarmacia|perfume|perro"
    r")\b"
)
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
CREAM_CHEESE_NEUTRAL_TOKENS = {
    "crema",
    "mascarpone",
    "philadelphia",
    "queso",
    "untar",
    "untable",
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

_RUNTIME_CACHE_LOCK = threading.Lock()
NON_SHOPPING_INGREDIENTS = {"agua"}


class RecipeDraftPlan(BaseModel):
    titulo: str = ""
    personas: int | None = None
    ingredientes: list[str] = Field(default_factory=list)


class RecipeFinalPlan(BaseModel):
    titulo: str = ""
    ingredientes_usados: list[str] = Field(default_factory=list)
    receta: str = ""


def _get_reasoning_effort() -> str | None:
    effort = os.getenv("RECETONA_REASONING_EFFORT", "").strip().lower()
    if not effort:
        return "none"
    return effort


def _get_reasoning_options() -> dict[str, str] | None:
    effort = _get_reasoning_effort()
    if not effort:
        return None
    return {"effort": effort}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_runtime_base_dir() -> Path:
    override = os.getenv(RUNTIME_BASE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return BASE_DIR


def get_runtime_rag_cache_dir() -> Path:
    override = os.getenv(RUNTIME_CACHE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return get_runtime_base_dir() / "rag_cache"


def get_runtime_rag_cache_paths() -> dict[str, Path]:
    cache_dir = get_runtime_rag_cache_dir()
    return {name: cache_dir / name for name in REQUIRED_RAG_CACHE_FILES}


def _get_rag_cache_s3_config() -> tuple[str, str]:
    bucket_name = os.getenv(RAG_CACHE_S3_BUCKET_ENV, "").strip()
    prefix = os.getenv(RAG_CACHE_S3_PREFIX_ENV, "").strip().strip("/")
    return bucket_name, prefix


def runtime_rag_cache_status() -> dict[str, Any]:
    cache_dir = get_runtime_rag_cache_dir()
    cache_paths = get_runtime_rag_cache_paths()
    missing_files = [
        name for name, path in cache_paths.items() if not path.exists()
    ]
    bucket_name, prefix = _get_rag_cache_s3_config()

    return {
        "cache_dir": cache_dir,
        "cache_paths": cache_paths,
        "missing_files": missing_files,
        "remote_configured": bool(bucket_name),
        "s3_bucket": bucket_name or None,
        "s3_prefix": prefix or None,
    }


def ensure_runtime_rag_cache(
    *,
    raise_on_error: bool,
    eager_download: bool,
) -> dict[str, Any]:
    status = runtime_rag_cache_status()
    if not status["missing_files"]:
        return status

    if not eager_download or not status["remote_configured"]:
        if raise_on_error and not status["remote_configured"]:
            raise RuntimeError(
                "Falta cache RAG precalculada y no hay bucket S3 configurado."
            )
        return status

    cache_dir = status["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket_name = str(status["s3_bucket"])
    prefix = str(status["s3_prefix"] or "").strip("/")

    with _RUNTIME_CACHE_LOCK:
        status = runtime_rag_cache_status()
        if status["missing_files"]:
            try:
                import boto3

                region_name = (
                    os.getenv("AWS_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "eu-west-1"
                )
                s3_client = boto3.client("s3", region_name=region_name)

                for filename in status["missing_files"]:
                    destination_path = cache_dir / filename
                    temporary_path = cache_dir / f".{filename}.tmp"
                    object_key = f"{prefix}/{filename}" if prefix else filename
                    logging.info(
                        "Descargando cache RAG desde s3://%s/%s a %s",
                        bucket_name,
                        object_key,
                        destination_path,
                    )
                    try:
                        s3_client.download_file(
                            bucket_name,
                            object_key,
                            str(temporary_path),
                        )
                        temporary_path.replace(destination_path)
                    finally:
                        if temporary_path.exists():
                            temporary_path.unlink()
            except Exception as exc:
                if raise_on_error:
                    raise RuntimeError(
                        "No se pudo descargar la cache RAG desde S3."
                    ) from exc

        status = runtime_rag_cache_status()

    if raise_on_error and status["missing_files"]:
        joined_missing = ", ".join(status["missing_files"])
        raise RuntimeError(
            "Faltan artefactos de cache RAG tras la descarga: "
            f"{joined_missing}."
        )

    return status


def _extract_code_cells(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            code_cells.append(source)
    return code_cells


def _select_runtime_cells(code_cells: list[str]) -> list[str]:
    selected = []
    markers = [
        "EMBED_MODEL =",
        "def _clean(v):",
        "client = OpenAI",
        "embeddings = ensure_embeddings(chunks)",
        "def ask_agent(",
    ]
    for marker in markers:
        match = next((c for c in code_cells if marker in c), None)
        if not match:
            raise RuntimeError(
                f"No se encontro la celda requerida con marcador: {marker}"
            )
        selected.append(match)
    return selected


def _normalize_matching_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _tokenize_normalized_text(value: Any) -> set[str]:
    normalized = _normalize_matching_text(value)
    return set(re.findall(r"[a-z0-9]+", normalized))


def _candidate_has_non_recipe_taxonomy(row: Any) -> bool:
    product_name_text = _normalize_matching_text(row.get("product_name"))
    taxonomy_text = " ".join(
        _normalize_matching_text(row.get(column_name))
        for column_name in ("category", "subcategory", "subsubcategory")
    )
    haystack = " ".join(
        part for part in (product_name_text, taxonomy_text) if part
    )
    return bool(NON_RECIPE_TAXONOMY_PATTERN.search(haystack))


def _candidate_is_incompatible_for_ingredient(
    ingredient: str,
    row: Any,
    recipe_query: str | None = None,
) -> bool:
    ingredient_text = _normalize_matching_text(ingredient)
    ingredient_tokens = _tokenize_normalized_text(ingredient_text)

    if not ingredient_tokens:
        return False

    product_name_text = _normalize_matching_text(row.get("product_name"))
    taxonomy_text = " ".join(
        _normalize_matching_text(row.get(column_name))
        for column_name in ("category", "subcategory", "subsubcategory")
    )
    haystack = " ".join(
        part for part in (product_name_text, taxonomy_text) if part
    )
    product_name_tokens = _tokenize_normalized_text(product_name_text)
    haystack_tokens = _tokenize_normalized_text(haystack)
    recipe_query_text = _normalize_matching_text(recipe_query)

    if _candidate_has_non_recipe_taxonomy(row):
        return True

    if (
        ingredient_tokens & RAW_NUT_INGREDIENT_TOKENS
        and PREPARED_FOOD_PATTERN.search(haystack)
    ):
        return True

    if CARROT_INGREDIENT_PATTERN.search(ingredient_text):
        if not (product_name_tokens & CARROT_NAME_TOKENS):
            return True

    if GARLIC_INGREDIENT_PATTERN.search(ingredient_text):
        if not (product_name_tokens & GARLIC_NAME_TOKENS):
            return True
        if product_name_tokens & GARLIC_PREPARED_FOOD_TOKENS:
            return True

    if CRUSHED_TOMATO_INGREDIENT_PATTERN.search(ingredient_text):
        if not {"tomate", "triturado"} <= product_name_tokens:
            return True

    if GELATIN_INGREDIENT_PATTERN.search(ingredient_text):
        if "gelatina" not in product_name_tokens:
            return True
        if product_name_tokens & FLAVORED_GELATIN_TOKENS:
            return True
        if "postres" in haystack_tokens:
            return True

    if CREAM_CHEESE_INGREDIENT_PATTERN.search(ingredient_text):
        if not (
            product_name_tokens & CREAM_CHEESE_NEUTRAL_TOKENS
            or haystack_tokens & {"untable", "untar"}
        ):
            return True
        if product_name_tokens & CREAM_CHEESE_FORBIDDEN_TOKENS:
            return True

    if BEEF_BROTH_INGREDIENT_PATTERN.search(ingredient_text):
        if "caldo" not in product_name_tokens:
            return True
        if not (
            product_name_tokens
            & {"carne", "cocido", "res", "ternera", "vacuno"}
        ):
            return True

    if PAPRIKA_INGREDIENT_PATTERN.search(ingredient_text):
        if "pimenton" not in product_name_tokens:
            return True

    if BEEF_INGREDIENT_PATTERN.search(ingredient_text):
        if not (
            product_name_tokens & BEEF_NAME_TOKENS
            or haystack_tokens & BEEF_NAME_TOKENS
        ):
            return True
        if (
            recipe_query_text
            and STEW_RECIPE_PATTERN.search(recipe_query_text)
            and product_name_tokens & BEEF_STEW_FORBIDDEN_TOKENS
        ):
            return True

    return False


def _filter_incompatible_ingredient_candidates(
    df_hits: Any,
    *,
    recipe_query: str | None = None,
) -> Any:
    if df_hits is None or getattr(df_hits, "empty", True):
        return df_hits

    try:
        ingredient = str(df_hits["ingredient"].iloc[0]).strip()
    except Exception:
        return df_hits

    if not ingredient:
        return df_hits

    compatible_mask = df_hits.apply(
        lambda row: not _candidate_is_incompatible_for_ingredient(
            ingredient,
            row,
            recipe_query=recipe_query,
        ),
        axis=1,
    )

    if bool(compatible_mask.any()):
        return df_hits.loc[compatible_mask].reset_index(drop=True)

    non_recipe_mask = df_hits.apply(
        _candidate_has_non_recipe_taxonomy,
        axis=1,
    )
    if bool((~non_recipe_mask).any()):
        return df_hits.loc[~non_recipe_mask].reset_index(drop=True)

    return df_hits.iloc[0:0].copy()


def _filter_non_shopping_ingredients(
    ingredients: list[str],
) -> list[str]:
    filtered_ingredients: list[str] = []

    for ingredient in ingredients:
        normalized_ingredient = _normalize_matching_text(ingredient)
        if normalized_ingredient in NON_SHOPPING_INGREDIENTS:
            continue
        filtered_ingredients.append(ingredient)

    return filtered_ingredients


def _filter_cost_plan_catalog_plausibility(
    plan_df: Any,
    *,
    recipe_query: str,
) -> Any:
    if plan_df is None or getattr(plan_df, "empty", True):
        return plan_df

    compatible_rows = []
    for _, row in plan_df.iterrows():
        ingredient = _normalize_matching_text(row.get("ingredient"))
        if not ingredient or ingredient in NON_SHOPPING_INGREDIENTS:
            continue
        if _candidate_is_incompatible_for_ingredient(
            ingredient,
            row,
            recipe_query=recipe_query,
        ):
            continue
        compatible_rows.append(row.to_dict())

    if not compatible_rows:
        return plan_df.iloc[0:0].copy()

    return plan_df.__class__(compatible_rows)


def _collect_missing_recipe_ingredients(
    plan_df: Any,
    *,
    max_items: int = 8,
) -> list[str]:
    if plan_df is None or getattr(plan_df, "empty", True):
        return []

    missing_ingredients: list[str] = []
    seen: set[str] = set()

    for _, row in plan_df.iterrows():
        ingredient = str(row.get("ingredient") or "").strip()
        product_name = str(row.get("product_name") or "").strip()
        normalized_ingredient = _normalize_matching_text(ingredient)

        if not ingredient or not normalized_ingredient:
            continue

        if product_name and product_name.lower() != "nan":
            continue

        if normalized_ingredient in seen:
            continue

        seen.add(normalized_ingredient)
        missing_ingredients.append(ingredient)
        if len(missing_ingredients) >= max_items:
            break

    return missing_ingredients


def _extract_parsed_response_payload(response: Any) -> Any | None:
    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue
        for content_item in getattr(output, "content", []):
            if getattr(content_item, "type", None) == "refusal":
                return None
            parsed_payload = getattr(content_item, "parsed", None)
            if parsed_payload is not None:
                return parsed_payload
    return None


def _call_structured_response(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    text_format: type[BaseModel],
    max_output_tokens: int,
) -> tuple[Any, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
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
        "text_format": text_format,
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
    }
    reasoning = _get_reasoning_options()
    if reasoning:
        request_kwargs["reasoning"] = reasoning

    response = client.responses.parse(**request_kwargs)
    parsed_payload = _extract_parsed_response_payload(response)
    if parsed_payload is None:
        raise RuntimeError(
            "La respuesta estructurada del modelo no es valida."
        )
    return parsed_payload, response


def _canonicalize_recipe_ingredient_name(value: Any) -> str:
    normalized_value = _normalize_matching_text(value)
    normalized_value = re.sub(r"\s+", " ", normalized_value).strip()
    return normalized_value.strip(" ,.;:-")


def _dedupe_recipe_ingredient_names(values: list[Any]) -> list[str]:
    deduped_values: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized_value = _canonicalize_recipe_ingredient_name(value)
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        deduped_values.append(normalized_value)

    return deduped_values


def _ingredient_matches_used_ingredient(
    ingredient_name: Any,
    used_ingredients: list[str],
) -> bool:
    normalized_ingredient = _canonicalize_recipe_ingredient_name(
        ingredient_name
    )
    if not normalized_ingredient:
        return False

    ingredient_tokens = set(re.findall(r"[a-z0-9]+", normalized_ingredient))
    for raw_used_ingredient in used_ingredients:
        normalized_used_ingredient = _canonicalize_recipe_ingredient_name(
            raw_used_ingredient
        )
        if not normalized_used_ingredient:
            continue
        if normalized_used_ingredient == normalized_ingredient:
            return True

        used_tokens = set(re.findall(r"[a-z0-9]+", normalized_used_ingredient))
        if len(used_tokens) >= 2 and used_tokens <= ingredient_tokens:
            return True
        if len(ingredient_tokens) >= 2 and ingredient_tokens <= used_tokens:
            return True

    return False


def _filter_cost_plan_to_used_ingredients(
    plan_df: Any,
    *,
    used_ingredients: list[str],
) -> Any:
    if plan_df is None or getattr(plan_df, "empty", True):
        return plan_df

    normalized_used_ingredients = _dedupe_recipe_ingredient_names(
        list(used_ingredients)
    )
    if not normalized_used_ingredients:
        return plan_df

    filtered_mask = plan_df["ingredient"].apply(
        lambda ingredient_name: _ingredient_matches_used_ingredient(
            ingredient_name,
            normalized_used_ingredients,
        )
    )

    if bool(filtered_mask.any()):
        return plan_df.loc[filtered_mask].reset_index(drop=True)

    return plan_df


def _summarize_cost_plan(
    plan_df: Any,
    *,
    servings: int,
) -> dict[str, Any]:
    if plan_df is None or getattr(plan_df, "empty", True):
        return {
            "servings": servings,
            "total_purchase_eur": 0.0,
            "total_escandallo_eur": 0.0,
            "missing_purchase_ingredients": [],
            "missing_escandallo_ingredients": [],
        }

    purchase_cost_series = plan_df["purchase_cost_eur"].dropna()
    escandallo_cost_series = plan_df["escandallo_cost_eur"].dropna()
    missing_purchase = plan_df[plan_df["purchase_cost_eur"].isna()][
        "ingredient"
    ].tolist()
    missing_escandallo = plan_df[plan_df["escandallo_cost_eur"].isna()][
        "ingredient"
    ].tolist()

    return {
        "servings": servings,
        "total_purchase_eur": (
            float(purchase_cost_series.sum())
            if not purchase_cost_series.empty
            else 0.0
        ),
        "total_escandallo_eur": (
            float(escandallo_cost_series.sum())
            if not escandallo_cost_series.empty
            else 0.0
        ),
        "missing_purchase_ingredients": missing_purchase,
        "missing_escandallo_ingredients": missing_escandallo,
    }


def _build_recipe_draft_prompt(
    *,
    query: str,
    servings: int,
    inferred_ingredients: list[str],
) -> str:
    hints_text = ""
    cleaned_hints = _dedupe_recipe_ingredient_names(inferred_ingredients)
    if cleaned_hints:
        hints_text = (
            "Pistas del recuperador actual, usalas solo si tienen sentido:\n"
            + "\n".join(f"- {ingredient}" for ingredient in cleaned_hints)
            + "\n\n"
        )

    return (
        f"Solicitud del usuario: {query}\n"
        f"Personas objetivo: {servings}\n\n"
        f"{hints_text}"
        "Devuelve un plan base de receta, no la respuesta final.\n"
        "Reglas:\n"
        "- Genera un titulo corto y natural en espanol.\n"
        "- Incluye ingredientes nucleares, buscables en catalogo, sin marca.\n"
        "- Excluye guarniciones opcionales, sugerencias de servido y agua "
        "salvo que sea esencial.\n"
        "- Prioriza ingredientes reales del plato, no utensilios ni tecnicas.\n"
        "- Usa nombres cortos como 'pollo', 'curry', 'leche de coco', "
        "'cebolla', 'ajo' o 'queso crema'.\n"
        "- No inventes productos ni precios.\n"
        "- Si la solicitud nombra un ingrediente principal, debe aparecer "
        "en la lista salvo que la receta no lo necesite de verdad."
    )


def _build_available_products_prompt_text(
    plan_df: Any,
    *,
    max_items: int = 18,
) -> str:
    if plan_df is None or getattr(plan_df, "empty", True):
        return "No hay productos concretos disponibles."

    lines: list[str] = []
    for _, row in plan_df.head(max_items).iterrows():
        ingredient = str(row.get("ingredient") or "").strip()
        product_name = str(row.get("product_name") or "").strip()
        if not product_name or product_name.lower() == "nan":
            continue

        quantity_text = "cantidad N/D"
        required_qty = row.get("required_qty")
        required_unit = row.get("required_unit")
        try:
            quantity_text = (
                f"{required_qty:g} {required_unit}"
                if isinstance(required_qty, (int, float))
                else f"{required_qty} {required_unit}"
            ).strip()
        except Exception:
            quantity_text = f"{required_qty} {required_unit}".strip()

        price_value = row.get("purchase_cost_eur")
        if isinstance(price_value, (int, float)):
            price_text = f"{price_value:.2f} €"
        else:
            price_text = "N/D"

        lines.append(
            "- "
            f"etiqueta={ingredient}; "
            f"producto={product_name}; "
            f"cantidad={quantity_text}; "
            f"compra_total={price_text}"
        )

    return (
        "\n".join(lines)
        if lines
        else "No hay productos concretos disponibles."
    )


def _build_final_recipe_prompt(
    *,
    query: str,
    draft_plan: RecipeDraftPlan,
    plan_df: Any,
    max_chars: int,
) -> str:
    available_products_text = _build_available_products_prompt_text(plan_df)
    missing_ingredients = _collect_missing_recipe_ingredients(plan_df)
    available_ingredients = {
        _normalize_matching_text(row.get("ingredient"))
        for _, row in (
            plan_df.iterrows()
            if plan_df is not None and not getattr(plan_df, "empty", True)
            else []
        )
        if _normalize_matching_text(row.get("ingredient"))
    }
    for ingredient in draft_plan.ingredientes:
        normalized_ingredient = _normalize_matching_text(ingredient)
        if (
            normalized_ingredient
            and normalized_ingredient not in available_ingredients
            and ingredient not in missing_ingredients
        ):
            missing_ingredients.append(ingredient)
    missing_text = "Ninguno"
    if missing_ingredients:
        missing_text = "\n".join(
            f"- {ingredient}" for ingredient in missing_ingredients
        )

    desired_ingredients_text = "\n".join(
        f"- {ingredient}" for ingredient in draft_plan.ingredientes
    )
    if not desired_ingredients_text:
        desired_ingredients_text = "- Sin ingredientes sugeridos"

    return (
        f"Solicitud del usuario: {query}\n"
        f"Titulo base: {draft_plan.titulo or 'Receta'}\n"
        f"Personas: {draft_plan.personas or 4}\n\n"
        "Ingredientes deseados del plato:\n"
        f"{desired_ingredients_text}\n\n"
        "Productos disponibles seleccionados desde el catalogo:\n"
        f"{available_products_text}\n\n"
        "Ingredientes deseados sin producto exacto disponible:\n"
        f"{missing_text}\n\n"
        "Escribe la receta final adaptada al catalogo disponible.\n"
        "Reglas estrictas:\n"
        f"- Maximo {max_chars} caracteres en receta.\n"
        "- Usa solo productos disponibles de la lista anterior.\n"
        "- Si falta un ingrediente habitual, adapta la receta a lo que si "
        "esta disponible y no lo menciones como si existiera.\n"
        "- No renombres un producto disponible como si fuera otro distinto.\n"
        "- No conviertas mantequilla en 'mantequilla de cacao'.\n"
        "- No conviertas una gelatina con sabor, un postre gelificado o un "
        "preparado saborizado en gelatina neutra, en hojas o en laminas.\n"
        "- No conviertas un queso con sabor especifico, como camembert, en "
        "queso crema neutro.\n"
        "- La lista ingredientes_usados debe contener solo etiquetas "
        "exactas del campo etiqueta=... de los productos disponibles que "
        "realmente usas en la receta final.\n"
        "- No incluyas guarniciones opcionales ni sugerencias de servido.\n"
        "- No inventes precios ni productos.\n"
        "- Mantén el plato reconocible y viable.\n"
        "- Devuelve la receta final en espanol con pasos claros."
    )


def _build_recipe_generation_prompt(
    query: str,
    plan_df: Any,
    *,
    max_chars: int,
    max_items: int = 14,
) -> str:
    available_lines: list[str] = []
    if plan_df is not None and not getattr(plan_df, "empty", True):
        for _, row in plan_df.head(max_items).iterrows():
            ingredient = str(row.get("ingredient") or "").strip()
            product_name = str(row.get("product_name") or "").strip()
            if not product_name or product_name.lower() == "nan":
                continue
            if ingredient:
                available_lines.append(f"- {ingredient}: {product_name}")
            else:
                available_lines.append(f"- {product_name}")

    available_text = "\n".join(available_lines)
    if not available_text:
        available_text = "No hay productos concretos disponibles."

    missing_ingredients = _collect_missing_recipe_ingredients(plan_df)
    missing_text = ""
    if missing_ingredients:
        missing_text = (
            "Ingredientes solicitados sin producto exacto disponible:\n"
            + "\n".join(
                f"- {ingredient}" for ingredient in missing_ingredients
            )
            + "\n\n"
        )

    return (
        f"Escribe una receta breve en español para esta solicitud: {query}.\n\n"
        f"Productos disponibles en Mercadona para esta receta:\n"
        f"{available_text}\n\n"
        f"{missing_text}"
        f"Reglas estrictas:\n"
        f"- Máximo {max_chars} caracteres.\n"
        f"- Usa solo los productos disponibles listados arriba.\n"
        f"- Si falta un ingrediente habitual o no tiene producto exacto, "
        f"adapta la receta a lo disponible en lugar de inventarlo.\n"
        f"- No menciones ingredientes que no aparezcan en la lista de "
        f"productos disponibles.\n"
        f"- Si una sustitución no encaja, omite ese ingrediente y ajusta "
        f"los pasos para que la receta siga siendo viable.\n"
        f"- Respeta el formato y tipo literal del producto disponible. "
        f"No conviertas un producto en otro formato distinto ni inventes "
        f"versiones neutras, en hojas o equivalentes si no aparecen "
        f"arriba.\n"
        f"- Mantén el plato reconocible y realista con los ingredientes "
        f"disponibles.\n"
        f"- Texto corrido con pasos claros (sin tablas).\n"
        f"- No inventes precios.\n"
        f"- Devuelve solo el texto de la receta."
    )


def _run_two_phase_recipe_pipeline(
    namespace: dict[str, Any],
    *,
    query: str,
    top_k: int,
    model: str,
    retrieval_mode: str,
    alpha: float,
    recipe_mode: str,
    use_ingredient_tool: bool,
    candidates_per_ingredient: int,
) -> dict[str, Any]:
    retrieve = namespace["retrieve"]
    parse_servings = namespace["parse_servings"]
    tool_get_products_for_ingredients = namespace[
        "tool_get_products_for_ingredients"
    ]
    build_recipe_cost_plan = namespace["build_recipe_cost_plan"]
    format_ingredient_catalog_text = namespace[
        "format_ingredient_catalog_text"
    ]
    format_cost_plan_text = namespace["format_cost_plan_text"]
    build_block1_ingredients_mercadona = namespace[
        "build_block1_ingredients_mercadona"
    ]
    tool_get_total_purchase_price = namespace["tool_get_total_purchase_price"]
    compose_structured_answer = namespace["compose_structured_answer"]
    catalog_preview = namespace["_catalog_preview"]
    remove_redundant_ingredients = namespace["_remove_redundant_ingredients"]
    build_block3_recipe_text = namespace["build_block3_recipe_text"]
    client = namespace["client"]

    hits, subqueries, inferred_ingredients = retrieve(
        query,
        top_k=top_k,
        mode=retrieval_mode,
        alpha=alpha,
        recipe_mode=recipe_mode,
    )
    servings = parse_servings(query, default=4)

    draft_plan, raw_draft_response = _call_structured_response(
        client=client,
        model=model,
        system_prompt=(
            "Eres el planificador culinario de RecetONA. Tu trabajo es "
            "extraer un plan base de ingredientes nucleares antes de "
            "consultar el catalogo de Mercadona."
        ),
        user_prompt=_build_recipe_draft_prompt(
            query=query,
            servings=servings,
            inferred_ingredients=inferred_ingredients,
        ),
        text_format=RecipeDraftPlan,
        max_output_tokens=250,
    )
    cleaned_draft_ingredients = _dedupe_recipe_ingredient_names(
        list(getattr(draft_plan, "ingredientes", []))
    )
    if not cleaned_draft_ingredients:
        cleaned_draft_ingredients = _dedupe_recipe_ingredient_names(
            list(inferred_ingredients)
        )
    cleaned_draft_ingredients = remove_redundant_ingredients(
        cleaned_draft_ingredients
    )
    cleaned_draft_ingredients = _filter_non_shopping_ingredients(
        cleaned_draft_ingredients
    )
    if not cleaned_draft_ingredients:
        raise RuntimeError(
            "No se pudieron inferir ingredientes nucleares para la receta."
        )

    ingredient_catalog = {}
    ingredient_catalog_text = ""
    cost_plan_df = None
    cost_summary = _summarize_cost_plan(None, servings=servings)
    cost_plan_text = "Sin plan de costes."

    if use_ingredient_tool:
        ingredient_catalog = tool_get_products_for_ingredients(
            cleaned_draft_ingredients,
            per_ingredient=candidates_per_ingredient,
            alpha=0.35,
            recipe_query=query,
        )
        ingredient_catalog_text = format_ingredient_catalog_text(
            ingredient_catalog,
            max_items=6,
        )
        cost_plan_df, cost_summary = build_recipe_cost_plan(
            ingredient_catalog=ingredient_catalog,
            ingredients=cleaned_draft_ingredients,
            query=query,
        )
        cost_plan_df = _filter_cost_plan_catalog_plausibility(
            cost_plan_df,
            recipe_query=query,
        )
        cost_summary["servings"] = int(
            getattr(draft_plan, "personas", None) or servings
        )
        cost_summary = _summarize_cost_plan(
            cost_plan_df,
            servings=cost_summary["servings"],
        )
        cost_plan_text = format_cost_plan_text(cost_plan_df, cost_summary)

    final_plan, raw_final_response = _call_structured_response(
        client=client,
        model=model,
        system_prompt=(
            "Eres el cocinero final de RecetONA. Debes redactar una receta "
            "usando solo los productos del catalogo ya seleccionados."
        ),
        user_prompt=_build_final_recipe_prompt(
            query=query,
            draft_plan=RecipeDraftPlan(
                titulo=str(getattr(draft_plan, "titulo", "") or "").strip(),
                personas=int(
                    getattr(draft_plan, "personas", None) or servings
                ),
                ingredientes=cleaned_draft_ingredients,
            ),
            plan_df=cost_plan_df,
            max_chars=1000,
        ),
        text_format=RecipeFinalPlan,
        max_output_tokens=650,
    )

    recipe_text = str(getattr(final_plan, "receta", "") or "").strip()
    if not recipe_text:
        recipe_text, raw_final_response = build_block3_recipe_text(
            query,
            cost_plan_df,
            model=model,
            max_chars=1000,
        )
    elif len(recipe_text) > 1000:
        recipe_text = recipe_text[:999].rstrip() + "…"

    used_ingredients = _dedupe_recipe_ingredient_names(
        list(getattr(final_plan, "ingredientes_usados", []))
    )
    if used_ingredients:
        filtered_cost_plan_df = _filter_cost_plan_to_used_ingredients(
            cost_plan_df,
            used_ingredients=used_ingredients,
        )
    else:
        filtered_cost_plan_df = cost_plan_df

    final_servings = int(getattr(draft_plan, "personas", None) or servings)
    final_cost_summary = _summarize_cost_plan(
        filtered_cost_plan_df,
        servings=final_servings,
    )
    final_cost_plan_text = format_cost_plan_text(
        filtered_cost_plan_df,
        final_cost_summary,
    )
    block_1 = build_block1_ingredients_mercadona(filtered_cost_plan_df)
    block_2 = tool_get_total_purchase_price(final_cost_summary)
    structured_answer = compose_structured_answer(
        block_1, block_2, recipe_text
    )

    hit_cols = [
        "product_id",
        "product_name",
        "category",
        "price_unit",
        "unit_size",
        "size_format",
        "score",
    ]
    available_hit_cols = [
        column_name for column_name in hit_cols if column_name in hits.columns
    ]
    for extra_column in ("score_semantic", "score_lexical"):
        if extra_column in hits.columns:
            available_hit_cols.append(extra_column)

    merged_inferred_ingredients = _dedupe_recipe_ingredient_names(
        cleaned_draft_ingredients + used_ingredients
    )

    return {
        "answer": structured_answer,
        "block_1": block_1,
        "block_2": block_2,
        "block_3": recipe_text,
        "hits": (
            hits[available_hit_cols].copy()
            if available_hit_cols
            else hits.copy()
        ),
        "subqueries": subqueries,
        "inferred_ingredients": merged_inferred_ingredients,
        "ingredient_catalog": ingredient_catalog,
        "ingredient_catalog_preview": catalog_preview(ingredient_catalog, n=3),
        "cost_plan": filtered_cost_plan_df,
        "cost_summary": final_cost_summary,
        "cost_plan_text": final_cost_plan_text,
        "ingredient_catalog_text": ingredient_catalog_text,
        "raw_response": {
            "draft": raw_draft_response,
            "final": raw_final_response,
        },
    }


def _patch_notebook_runtime(namespace: dict[str, Any]) -> None:
    original_retrieve = namespace.get("retrieve_products_for_ingredient")
    if callable(original_retrieve):

        def _patched_retrieve_products_for_ingredient(
            ingredient, top_n=10, alpha=0.35, recipe_query=None
        ):
            df_hits = original_retrieve(
                ingredient,
                top_n=top_n,
                alpha=alpha,
                recipe_query=recipe_query,
            )
            return _filter_incompatible_ingredient_candidates(
                df_hits,
                recipe_query=recipe_query,
            )

        namespace["retrieve_products_for_ingredient"] = (
            _patched_retrieve_products_for_ingredient
        )

    original_choose = namespace.get("_choose_best_candidate")
    if callable(original_choose):

        def _patched_choose_best_candidate(
            df_hits,
            required_qty,
            required_unit,
            requirement_source="fallback",
        ):
            filtered_hits = _filter_incompatible_ingredient_candidates(df_hits)
            return original_choose(
                filtered_hits,
                required_qty,
                required_unit,
                requirement_source,
            )

        namespace["_choose_best_candidate"] = _patched_choose_best_candidate

    original_build_block3 = namespace.get("build_block3_recipe_text")
    client = namespace.get("client")
    if callable(original_build_block3) and client is not None:

        def _patched_build_block3_recipe_text(
            query,
            plan_df,
            model=None,
            max_chars=1000,
        ):
            prompt = _build_recipe_generation_prompt(
                query,
                plan_df,
                max_chars=max_chars,
            )
            raw_resp = None
            text = ""
            resolved_model = model or namespace.get("CHAT_MODEL")

            try:
                request_kwargs: dict[str, Any] = {
                    "model": resolved_model,
                    "input": prompt,
                }
                reasoning = _get_reasoning_options()
                if reasoning:
                    request_kwargs["reasoning"] = reasoning
                raw_resp = client.responses.create(**request_kwargs)
                text = (raw_resp.output_text or "").strip()
            except Exception as exc:
                text = (
                    "No se pudo generar el texto de receta "
                    f"automáticamente ({exc})."
                )

            if len(text) > max_chars:
                text = text[: max_chars - 1].rstrip() + "…"

            return text, raw_resp

        namespace["build_block3_recipe_text"] = (
            _patched_build_block3_recipe_text
        )

    original_ask_agent = namespace.get("ask_agent")
    retrieve = namespace.get("retrieve")
    is_recipe_query = namespace.get("is_recipe_query")
    if (
        callable(original_ask_agent)
        and callable(retrieve)
        and callable(is_recipe_query)
        and client is not None
    ):

        def _patched_ask_agent(
            query,
            top_k=20,
            model=None,
            retrieval_mode="hybrid",
            alpha=0.65,
            recipe_mode="auto",
            use_ingredient_tool=True,
            candidates_per_ingredient=10,
        ):
            resolved_model = model or namespace.get("CHAT_MODEL")
            if not (
                use_ingredient_tool
                and is_recipe_query(query)
                and client is not None
            ):
                return original_ask_agent(
                    query,
                    top_k=top_k,
                    model=resolved_model,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    recipe_mode=recipe_mode,
                    use_ingredient_tool=use_ingredient_tool,
                    candidates_per_ingredient=candidates_per_ingredient,
                )

            try:
                return _run_two_phase_recipe_pipeline(
                    namespace,
                    query=query,
                    top_k=top_k,
                    model=resolved_model,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    recipe_mode=recipe_mode,
                    use_ingredient_tool=use_ingredient_tool,
                    candidates_per_ingredient=candidates_per_ingredient,
                )
            except Exception:
                logging.warning(
                    "Fallo el pipeline a dos fases de RecetONA. "
                    "Se usa el flujo heredado.",
                    exc_info=True,
                )
                return original_ask_agent(
                    query,
                    top_k=top_k,
                    model=resolved_model,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    recipe_mode=recipe_mode,
                    use_ingredient_tool=use_ingredient_tool,
                    candidates_per_ingredient=candidates_per_ingredient,
                )

        namespace["ask_agent"] = _patched_ask_agent


def build_notebook_runtime(notebook_path: Path) -> dict:
    cache_status = ensure_runtime_rag_cache(
        raise_on_error=False,
        eager_download=True,
    )
    cells = _extract_code_cells(notebook_path)
    runtime_cells = _select_runtime_cells(cells)

    code = "\n\n".join(runtime_cells)
    code = code.replace("/home/wencm/RecetONA", str(BASE_DIR))
    code = code.replace("/home/wencm/Alimentación", str(BASE_DIR))
    code = code.replace(
        "CACHE_DIR = BASE_DIR / 'rag_cache'",
        f"CACHE_DIR = Path({str(cache_status['cache_dir'])!r})",
    )

    namespace: dict = {"__name__": "__recetona_notebook_runtime__"}
    exec(compile(code, str(notebook_path), "exec"), namespace, namespace)
    _patch_notebook_runtime(namespace)
    resolved_embed_model = (
        os.getenv("RECETONA_EMBED_MODEL", "").strip()
        or str(namespace.get("EMBED_MODEL", "text-embedding-3-large")).strip()
        or "text-embedding-3-large"
    )
    resolved_chat_model = (
        os.getenv("RECETONA_CHAT_MODEL", "").strip()
        or str(namespace.get("CHAT_MODEL", "gpt-5.4-nano")).strip()
        or "gpt-5.4-nano"
    )
    namespace["EMBED_MODEL"] = resolved_embed_model
    namespace["CHAT_MODEL"] = resolved_chat_model
    logging.info(
        "Runtime notebook EMBED_MODEL=%s",
        namespace["EMBED_MODEL"],
    )
    logging.info(
        "Runtime notebook CHAT_MODEL=%s",
        namespace["CHAT_MODEL"],
    )
    if "ask_agent" not in namespace:
        raise RuntimeError("No se pudo cargar ask_agent desde el notebook.")
    return namespace


class NotebookRagService:
    def __init__(self, notebook_path: Path):
        self._lock = threading.Lock()
        self._ns = build_notebook_runtime(notebook_path)
        self._ask_agent = self._ns["ask_agent"]

    def ask(self, message: str) -> dict:
        with self._lock:
            result = self._ask_agent(
                message,
                top_k=35,
                retrieval_mode="hybrid",
                alpha=0.65,
                recipe_mode="auto",
                use_ingredient_tool=True,
                candidates_per_ingredient=12,
            )

        return {
            "answer": str(result.get("answer", "")).strip(),
            "block_1": str(result.get("block_1", "")).strip(),
            "block_2": str(result.get("block_2", "")).strip(),
            "block_3": str(result.get("block_3", "")).strip(),
            "cost_plan": result.get("cost_plan"),
            "cost_summary": result.get("cost_summary"),
            "inferred_ingredients": result.get("inferred_ingredients"),
        }


class LocalApiHandler(BaseHTTPRequestHandler):
    service: NotebookRagService | None = None

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True, "service": "recetona-local-rag"})
            return
        self._send_json(404, {"error": "Ruta no encontrada"})

    def do_POST(self):
        if self.path != "/chat":
            self._send_json(404, {"error": "Ruta no encontrada"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "JSON invalido"})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json(400, {"error": "Falta el campo 'message'"})
            return

        if self.service is None:
            self._send_json(503, {"error": "Servicio no inicializado"})
            return

        try:
            response = self.service.ask(message)
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(
                500,
                {
                    "error": f"Fallo ejecutando ask_agent: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Servidor local para conectar frontend con ask_agent del notebook."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    load_env_file(ENV_PATH)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"Falta OPENAI_API_KEY. Define la clave en {ENV_PATH} o en variables de entorno."
        )

    logging.info("Inicializando runtime desde notebook: %s", NOTEBOOK_PATH)
    service = NotebookRagService(NOTEBOOK_PATH)
    LocalApiHandler.service = service

    server = ThreadingHTTPServer((args.host, args.port), LocalApiHandler)
    logging.info("API local escuchando en http://%s:%d", args.host, args.port)
    logging.info("Endpoints: GET /health, POST /chat")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Cerrando servidor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
