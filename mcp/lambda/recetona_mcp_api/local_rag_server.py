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

_RUNTIME_CACHE_LOCK = threading.Lock()


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
                raw_resp = client.responses.create(
                    model=resolved_model,
                    input=prompt,
                )
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
