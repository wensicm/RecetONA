from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from starlette.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag_server import (
    ENV_PATH,
    NOTEBOOK_PATH,
    NotebookRagService,
    ensure_runtime_rag_cache,
    load_env_file,
)

MCP_NAME = "RecetONA"
RECIPE_WIDGET_URI = "ui://widget/recetona-recipe-v1.html"
CATALOG_CACHE_PATH = PROJECT_ROOT / "rag_cache" / "chunks.csv"
EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "rag_cache" / "embeddings.npy"
EXCEL_PATH = PROJECT_ROOT / "mercadona_data.xlsx"

_service: NotebookRagService | None = None
_catalog_dataframe: pd.DataFrame | None = None
_resolved_openai_api_key: str | None = None
_resolved_openai_api_key_source: str | None = None


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


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
        "query_recipe requiere caches RAG precalculadas en Lambda. "
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
) -> dict[str, Any]:
    ingredients = _normalize_cost_plan_rows(result.get("cost_plan"))
    cost_summary = result.get("cost_summary")
    if not isinstance(cost_summary, dict):
        cost_summary = {}

    return {
        "pregunta": pregunta,
        "productos_mercadona_exactos": [
            item["producto_mercadona"] for item in ingredients
        ],
        "ingredientes_mercadona": ingredients,
        "ingredientes_mercadona_texto_literal": str(
            result.get("block_1", "")
        ).strip(),
        "receta_y_pasos_texto_literal": str(result.get("block_3", "")).strip(),
        "respuesta_literal_mcp": str(result.get("answer", "")).strip(),
        "coste_total_compra_eur": _safe_float_or_none(
            cost_summary.get("total_purchase_eur")
        ),
        "coste_total_consumido_eur": _safe_float_or_none(
            cost_summary.get("total_escandallo_eur")
        ),
        "personas": cost_summary.get("servings"),
        "ingredientes_inferidos": result.get("inferred_ingredients") or [],
    }


def _recipe_widget_resource_meta() -> dict[str, Any]:
    return {
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


def _query_recipe_tool_meta() -> dict[str, Any]:
    return {
        "ui": {
            "resourceUri": RECIPE_WIDGET_URI,
        },
        "openai/outputTemplate": RECIPE_WIDGET_URI,
        "openai/toolInvocation/invoking": "Preparando receta...",
        "openai/toolInvocation/invoked": "Receta lista",
    }


def _build_recipe_widget_html() -> str:
    return """<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RecetONA</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f7f1e7;
        --surface: #fffaf2;
        --surface-strong: #fffdf8;
        --ink: #2f2419;
        --muted: #6f5a46;
        --line: rgba(108, 74, 43, 0.16);
        --accent: #9b4d12;
        --accent-soft: rgba(155, 77, 18, 0.1);
        --shadow: 0 18px 48px rgba(79, 48, 24, 0.12);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        background:
          radial-gradient(circle at top right, rgba(225, 157, 76, 0.2), transparent 34%),
          linear-gradient(180deg, #fff8ef 0%, var(--bg) 100%);
        color: var(--ink);
        font:
          15px/1.5 "Iowan Old Style", "Palatino Linotype", "Book Antiqua",
          Georgia, serif;
      }

      .shell {
        padding: 18px;
      }

      .panel {
        background: linear-gradient(180deg, var(--surface) 0%, var(--surface-strong) 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }

      .hero {
        padding: 20px 20px 16px;
        border-bottom: 1px solid var(--line);
        background:
          radial-gradient(circle at top left, rgba(155, 77, 18, 0.12), transparent 28%),
          linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(249, 241, 228, 0.98));
      }

      .eyebrow {
        margin: 0 0 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
      }

      h1 {
        margin: 0;
        font-size: 28px;
        line-height: 1.1;
      }

      .subtitle {
        margin: 8px 0 0;
        color: var(--muted);
      }

      .stats {
        display: grid;
        gap: 10px;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        padding: 16px 20px 0;
      }

      .stat {
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--accent-soft);
        border: 1px solid rgba(155, 77, 18, 0.08);
      }

      .stat-label {
        margin: 0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .stat-value {
        margin: 4px 0 0;
        font-size: 22px;
        line-height: 1.1;
      }

      .content {
        display: grid;
        gap: 18px;
        padding: 18px;
      }

      .section {
        padding: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.58);
      }

      .section h2 {
        margin: 0 0 12px;
        font-size: 18px;
        line-height: 1.2;
      }

      .ingredient-list,
      .step-list {
        display: grid;
        gap: 10px;
      }

      .ingredient-card {
        padding: 12px 14px;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: rgba(255, 252, 247, 0.92);
      }

      .ingredient-target {
        margin: 0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent);
      }

      .ingredient-name {
        margin: 4px 0 0;
        font-size: 18px;
        line-height: 1.2;
      }

      .ingredient-meta {
        margin: 8px 0 0;
        color: var(--muted);
      }

      .step {
        display: grid;
        grid-template-columns: 34px 1fr;
        gap: 12px;
        align-items: start;
      }

      .step-index {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background: var(--ink);
        color: #fff;
        font-weight: 700;
      }

      .step-text {
        margin: 0;
        padding-top: 4px;
      }

      details {
        border-radius: 14px;
        border: 1px dashed var(--line);
        background: rgba(255, 255, 255, 0.45);
        padding: 12px 14px;
      }

      summary {
        cursor: pointer;
        font-weight: 700;
      }

      pre {
        margin: 12px 0 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--muted);
        font:
          13px/1.5 ui-monospace, "SFMono-Regular", "SF Mono", Menlo,
          Consolas, monospace;
      }

      .empty {
        margin: 0;
        color: var(--muted);
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <article class="panel">
        <header class="hero">
          <p class="eyebrow">RecetONA</p>
          <h1 id="question">Preparando receta...</h1>
          <p class="subtitle">
            Lista de compra exacta y pasos generados con nombres literales de
            productos de Mercadona.
          </p>
          <div class="stats">
            <div class="stat">
              <p class="stat-label">Productos exactos</p>
              <p class="stat-value" id="product-count">-</p>
            </div>
            <div class="stat">
              <p class="stat-label">Personas</p>
              <p class="stat-value" id="servings">-</p>
            </div>
            <div class="stat">
              <p class="stat-label">Compra total</p>
              <p class="stat-value" id="purchase-cost">-</p>
            </div>
            <div class="stat">
              <p class="stat-label">Coste consumido</p>
              <p class="stat-value" id="consumed-cost">-</p>
            </div>
          </div>
        </header>

        <div class="content">
          <section class="section">
            <h2>Lista de compra exacta</h2>
            <div id="ingredients" class="ingredient-list"></div>
          </section>

          <section class="section">
            <h2>Pasos</h2>
            <div id="steps" class="step-list"></div>
          </section>

          <details>
            <summary>Texto literal devuelto por el MCP</summary>
            <pre id="literal-output"></pre>
          </details>
        </div>
      </article>
    </div>

    <script>
      const questionEl = document.getElementById("question");
      const productCountEl = document.getElementById("product-count");
      const servingsEl = document.getElementById("servings");
      const purchaseCostEl = document.getElementById("purchase-cost");
      const consumedCostEl = document.getElementById("consumed-cost");
      const ingredientsEl = document.getElementById("ingredients");
      const stepsEl = document.getElementById("steps");
      const literalOutputEl = document.getElementById("literal-output");

      function escapeHtml(value) {
        return String(value ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function formatMoney(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
          return "N/D";
        }
        return new Intl.NumberFormat(document.documentElement.lang || "es-ES", {
          style: "currency",
          currency: "EUR",
        }).format(Number(value));
      }

      function buildIngredientMeta(item) {
        const parts = [];
        if (item.cantidad_receta_valor !== null && item.cantidad_receta_valor !== undefined) {
          const amount = Number(item.cantidad_receta_valor);
          const amountText = Number.isInteger(amount)
            ? String(amount)
            : amount.toFixed(2).replace(/0+$/, "").replace(/\\.$/, "");
          if (item.cantidad_receta_unidad) {
            parts.push(amountText + " " + item.cantidad_receta_unidad);
          } else {
            parts.push(amountText);
          }
        }
        if (item.precio_envase_eur !== null && item.precio_envase_eur !== undefined) {
          parts.push("envase " + formatMoney(item.precio_envase_eur));
        }
        if (item.coste_compra_eur !== null && item.coste_compra_eur !== undefined) {
          parts.push("compra " + formatMoney(item.coste_compra_eur));
        }
        return parts.join(" · ");
      }

      function splitSteps(text) {
        const cleanText = String(text ?? "").trim();
        if (!cleanText) {
          return [];
        }
        return cleanText
          .split(/(?<=[.!?])\\s+/)
          .map((part) => part.trim())
          .filter(Boolean);
      }

      function getToolData() {
        return window.openai?.toolOutput || {};
      }

      function renderIngredients(items) {
        if (!Array.isArray(items) || items.length === 0) {
          ingredientsEl.innerHTML = '<p class="empty">No hay ingredientes estructurados.</p>';
          return;
        }

        ingredientsEl.innerHTML = items
          .map((item) => {
            return `
              <article class="ingredient-card">
                <p class="ingredient-target">${escapeHtml(item.ingrediente_objetivo || "Ingrediente")}</p>
                <p class="ingredient-name">${escapeHtml(item.producto_mercadona || "Producto sin nombre")}</p>
                <p class="ingredient-meta">${escapeHtml(buildIngredientMeta(item) || "Sin detalle de coste")}</p>
              </article>
            `;
          })
          .join("");
      }

      function renderSteps(text) {
        const steps = splitSteps(text);
        if (steps.length === 0) {
          stepsEl.innerHTML = '<p class="empty">No hay pasos disponibles.</p>';
          return;
        }

        stepsEl.innerHTML = steps
          .map((step, index) => {
            return `
              <div class="step">
                <span class="step-index">${index + 1}</span>
                <p class="step-text">${escapeHtml(step)}</p>
              </div>
            `;
          })
          .join("");
      }

      function render(data) {
        const payload = data && typeof data === "object" ? data : {};
        const ingredients = Array.isArray(payload.ingredientes_mercadona)
          ? payload.ingredientes_mercadona
          : [];

        questionEl.textContent = payload.pregunta || "Receta disponible";
        productCountEl.textContent = String(
          Array.isArray(payload.productos_mercadona_exactos)
            ? payload.productos_mercadona_exactos.length
            : ingredients.length
        );
        servingsEl.textContent = payload.personas || "-";
        purchaseCostEl.textContent = formatMoney(payload.coste_total_compra_eur);
        consumedCostEl.textContent = formatMoney(payload.coste_total_consumido_eur);

        renderIngredients(ingredients);
        renderSteps(payload.receta_y_pasos_texto_literal);

        literalOutputEl.textContent = String(payload.respuesta_literal_mcp || "").trim();
      }

      function handleToolResult(result) {
        if (result && typeof result === "object") {
          render(result.structuredContent || result);
          return;
        }
        render(getToolData());
      }

      window.addEventListener(
        "message",
        (event) => {
          if (event.source !== window.parent) {
            return;
          }
          const message = event.data;
          if (!message || message.jsonrpc !== "2.0") {
            return;
          }
          if (message.method !== "ui/notifications/tool-result") {
            return;
          }
          handleToolResult(message.params);
        },
        { passive: true }
      );

      window.addEventListener(
        "openai:set_globals",
        (event) => {
          handleToolResult(event.detail?.globals?.toolOutput || getToolData());
        },
        { passive: true }
      );

      render(getToolData());
    </script>
  </body>
</html>
"""


def _build_query_recipe_tool_result(
    payload: dict[str, Any],
) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    "Mostrando la receta exacta de Mercadona en el panel "
                    "adjunto."
                ),
            )
        ],
        structuredContent=payload,
        _meta={
            "recetona/widgetVersion": "recipe-v1",
        },
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
            "Servidor de recetas de Mercadona basado en RAG. "
            "Usa query_recipe para responder recetas con ingredientes y coste. "
            "Usa search/fetch para recuperar productos del catalogo."
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

    @mcp_server.tool(
        name="query_recipe",
        description=(
            "Devuelve una receta usando productos de Mercadona en formato "
            "estructurado. Al responder al usuario, conserva "
            "literalmente los nombres de `productos_mercadona_exactos` y "
            "de cada `producto_mercadona`; no los sustituyas por "
            "ingredientes genericos."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
        meta=_query_recipe_tool_meta(),
    )
    def query_recipe(pregunta: str) -> Any:
        pregunta_normalizada = str(pregunta).strip()
        if not pregunta_normalizada:
            raise ValueError("La pregunta no puede estar vacia.")

        result = _get_service().ask(pregunta_normalizada)
        payload = _build_query_recipe_payload(
            pregunta=pregunta_normalizada,
            result=result,
        )
        return _build_query_recipe_tool_result(payload)

    @mcp_server.tool(
        name="search",
        description=(
            "Busca productos del catalogo de Mercadona relacionados "
            "con una consulta."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def search(query: str, limit: int = 8) -> dict[str, list[dict[str, str]]]:
        normalized_query = str(query).strip()
        if not normalized_query:
            return {"results": []}

        bounded_limit = max(1, min(int(limit), 25))
        catalog_dataframe = _load_catalog()
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", _normalize(normalized_query))
            if len(token) > 1
        ]
        if not tokens:
            return {"results": []}

        scores = np.zeros(len(catalog_dataframe), dtype=np.int16)
        for token in tokens:
            scores += (
                catalog_dataframe["search_text"]
                .str.contains(re.escape(token), regex=True, na=False)
                .to_numpy(dtype=np.int16)
            )

        hit_indices = np.where(scores > 0)[0]
        if len(hit_indices) == 0:
            return {"results": []}

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

        return {"results": results}

    @mcp_server.tool(
        name="fetch",
        description=(
            "Devuelve el detalle completo de un producto del catalogo por id."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def fetch(id: str) -> dict[str, Any]:
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
            return _row_to_fetch_payload(row)

        if "row_idx" in catalog_dataframe.columns:
            row_index_as_text = catalog_dataframe["row_idx"].astype(str)
            row_index_mask = row_index_as_text == requested_id
            if row_index_mask.any():
                row = catalog_dataframe[row_index_mask].iloc[0]
                return _row_to_fetch_payload(row)

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
