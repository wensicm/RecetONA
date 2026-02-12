#!/usr/bin/env python3.12
"""MCP server for RecetONA recipe and catalog queries."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
LIB_DIR = BASE_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from local_rag_server import NOTEBOOK_PATH, ENV_PATH, NotebookRagService, load_env_file

MCP_NAME = "RecetONA"
CATALOG_CACHE_PATH = BASE_DIR / "rag_cache" / "chunks.csv"
EXCEL_PATH = BASE_DIR / "mercadona_data.xlsx"

mcp = FastMCP(
    name=MCP_NAME,
    instructions=(
        "Servidor de recetas de Mercadona basado en RAG. "
        "Usa query_recipe para responder recetas con ingredientes y coste. "
        "Usa search/fetch para recuperar productos del catalogo."
    ),
    # Use 0.0.0.0 so FastMCP does not auto-restrict Host headers to localhost.
    # This is required when exposing the server via HTTPS tunnels (e.g. trycloudflare).
    host="0.0.0.0",
    port=8788,
    streamable_http_path="/mcp",
    log_level="INFO",
)

_service: NotebookRagService | None = None
_catalog_df: pd.DataFrame | None = None


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def _fmt_product_id(value: Any) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value).strip()
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def _get_service() -> NotebookRagService:
    global _service
    if _service is not None:
        return _service

    load_env_file(ENV_PATH)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            f"Falta OPENAI_API_KEY. Define la clave en {ENV_PATH} o en variables de entorno."
        )

    _service = NotebookRagService(NOTEBOOK_PATH)
    return _service


def _load_catalog() -> pd.DataFrame:
    global _catalog_df
    if _catalog_df is not None:
        return _catalog_df

    if CATALOG_CACHE_PATH.exists():
        df = pd.read_csv(CATALOG_CACHE_PATH)
    elif EXCEL_PATH.exists():
        df = pd.read_excel(EXCEL_PATH)
        df["row_idx"] = np.arange(len(df), dtype=int)
    else:
        raise RuntimeError(f"No existe catalogo en {CATALOG_CACHE_PATH} ni {EXCEL_PATH}.")

    for col in ("product_name", "category", "subcategory", "subsubcategory", "ingredientes"):
        if col not in df.columns:
            df[col] = ""

    df["search_text"] = (
        df["product_name"].fillna("").astype(str)
        + " "
        + df["category"].fillna("").astype(str)
        + " "
        + df["subcategory"].fillna("").astype(str)
        + " "
        + df["subsubcategory"].fillna("").astype(str)
        + " "
        + df["ingredientes"].fillna("").astype(str)
    ).str.lower()

    _catalog_df = df
    return _catalog_df


def _row_to_fetch_payload(row: pd.Series) -> dict[str, Any]:
    product_id = _fmt_product_id(row.get("product_id", row.get("row_idx", "")))
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


@mcp.tool(
    name="query_recipe",
    description=(
        "Devuelve la respuesta final de receta ya lista para el usuario usando productos de Mercadona. "
        "La salida de esta tool debe mostrarse de forma literal, sin parafrasear ni renombrar productos."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def query_recipe(pregunta: str) -> str:
    """Consulta receta/productos con la logica del notebook."""
    pregunta = str(pregunta).strip()
    if not pregunta:
        raise ValueError("La pregunta no puede estar vacia.")

    result = _get_service().ask(pregunta)
    # Return only the final user-facing text to reduce model-side rewriting.
    return str(result["answer"]).strip()


@mcp.tool(
    name="search",
    description="Busca productos del catalogo de Mercadona relacionados con una consulta.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def search(query: str, limit: int = 8) -> dict[str, list[dict[str, str]]]:
    """Tool MCP estandar para descubrimiento de documentos/productos."""
    query = str(query).strip()
    if not query:
        return {"results": []}

    limit = max(1, min(int(limit), 25))
    df = _load_catalog()
    tokens = [t for t in re.findall(r"[a-z0-9]+", _normalize(query)) if len(t) > 1]
    if not tokens:
        return {"results": []}

    scores = np.zeros(len(df), dtype=np.int16)
    for tok in tokens:
        scores += df["search_text"].str.contains(re.escape(tok), regex=True, na=False).to_numpy(dtype=np.int16)

    hit_indices = np.where(scores > 0)[0]
    if len(hit_indices) == 0:
        return {"results": []}

    scored = df.iloc[hit_indices].copy()
    scored["score"] = scores[hit_indices]
    scored = scored.sort_values(by=["score", "price_unit"], ascending=[False, True]).head(limit)

    results = []
    for _, row in scored.iterrows():
        product_id = _fmt_product_id(row.get("product_id", row.get("row_idx", "")))
        title = str(row.get("product_name", f"Producto {product_id}")).strip()
        results.append(
            {
                "id": product_id,
                "title": title,
                "url": f"recetona://producto/{product_id}",
            }
        )

    return {"results": results}


@mcp.tool(
    name="fetch",
    description="Devuelve el detalle completo de un producto del catalogo por id.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def fetch(id: str) -> dict[str, Any]:
    """Tool MCP estandar para recuperar el contenido de un resultado search."""
    requested_id = str(id).strip()
    if not requested_id:
        raise ValueError("El id no puede estar vacio.")

    df = _load_catalog()

    # Try product_id exact match first.
    product_id_as_str = df.get("product_id", pd.Series(dtype=object)).map(_fmt_product_id)
    mask = product_id_as_str == requested_id
    if mask.any():
        row = df[mask].iloc[0]
        return _row_to_fetch_payload(row)

    # Fallback to row index.
    if "row_idx" in df.columns:
        row_idx_as_str = df["row_idx"].astype(str)
        mask = row_idx_as_str == requested_id
        if mask.any():
            row = df[mask].iloc[0]
            return _row_to_fetch_payload(row)

    raise ValueError(f"No existe producto con id '{requested_id}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Servidor MCP local de RecetONA para consultas de recetas."
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="Transporte MCP (recomendado: stdio para Codex local).",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
