from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

for candidate_path in (PROJECT_ROOT, SOURCE_DIR):
    resolved_path = str(candidate_path)
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

import recetona.mcp_app as mcp_app
from recetona.mcp_app import (
    RecipeWidgetPayload,
    build_streamable_http_app,
)


def test_health_route_reports_root_mcp_path():
    client = TestClient(build_streamable_http_app())

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "recetona-chatgpt-app"
    assert payload["mcp_path"] == "/"


def test_recipe_widget_path_resolves_external_html():
    widget_path = mcp_app._resolve_recipe_widget_html_path()

    assert widget_path.exists()
    assert widget_path.name == "recetona-widget.html"


def test_build_query_recipe_payload_returns_widget_model():
    result = {
        "answer": "respuesta literal completa",
        "block_1": "- 1 ud Chocolate negro fundir Hacendado (2,35€)",
        "block_3": "Mezcla y hornea.",
        "cost_plan": [
            {
                "ingredient": "chocolate negro",
                "required_qty": 1,
                "required_unit": "ud",
                "product_id": 12531,
                "product_name": "Chocolate negro fundir Hacendado",
                "unit_size": 0.2,
                "size_format": "kg",
                "price_unit": 2.35,
                "units_to_buy": 1,
                "purchase_cost_eur": 2.35,
                "escandallo_cost_eur": 2.35,
                "notes": "OK",
            }
        ],
        "cost_summary": {
            "servings": 4,
            "total_purchase_eur": 2.35,
            "total_escandallo_eur": 2.35,
        },
        "inferred_ingredients": ["chocolate negro"],
    }

    payload = mcp_app._build_query_recipe_payload(
        pregunta="tarta de chocolate",
        result=result,
    )

    assert isinstance(payload, RecipeWidgetPayload)
    assert payload.productos_mercadona_exactos == [
        "Chocolate negro fundir Hacendado"
    ]
    assert payload.ingredientes_mercadona[0].producto_mercadona == (
        "Chocolate negro fundir Hacendado"
    )


def test_query_recipe_data_tool_result_is_widget_ready():
    payload = RecipeWidgetPayload(
        pregunta="tarta de chocolate",
        productos_mercadona_exactos=["Chocolate negro fundir Hacendado"],
    )

    result = mcp_app._build_query_recipe_data_tool_result(payload)

    assert result.structuredContent == payload.model_dump(mode="json")
    assert result.meta is None
    assert "widget" in result.content[0].text.lower()


def test_render_recipe_tool_result_keeps_widget_meta():
    payload = RecipeWidgetPayload(
        pregunta="tarta de chocolate",
        productos_mercadona_exactos=["Chocolate negro fundir Hacendado"],
    )

    result = mcp_app._build_render_recipe_tool_result(payload)

    assert result.structuredContent == payload.model_dump(mode="json")
    assert result.meta["recetona/widgetVersion"] == "recipe-v2"
    assert "panel adjunto" in result.content[0].text


def test_render_recipe_tool_meta_links_widget_template():
    meta = mcp_app._render_recipe_tool_meta()

    assert meta["ui"]["resourceUri"] == mcp_app.RECIPE_WIDGET_URI
    assert meta["openai/outputTemplate"] == mcp_app.RECIPE_WIDGET_URI


def test_query_recipe_data_tool_meta_links_widget_template():
    meta = mcp_app._query_recipe_data_tool_meta()

    assert meta["ui"]["resourceUri"] == mcp_app.RECIPE_WIDGET_URI
    assert meta["openai/outputTemplate"] == mcp_app.RECIPE_WIDGET_URI


def test_build_query_recipe_payload_falls_back_to_catalog(monkeypatch):
    fake_catalog = pd.DataFrame(
        [
            {
                "product_id": 12531,
                "product_name": "Chocolate negro fundir Hacendado",
                "category": "Azucar, caramelos y chocolate",
                "subcategory": "Chocolate",
                "subsubcategory": "Chocolate negro",
                "price_unit": 2.35,
                "unit_size": 0.2,
                "size_format": "kg",
            },
            {
                "product_id": 84692,
                "product_name": "Croissant de mantequilla 26%",
                "category": "Panaderia y pasteleria",
                "subcategory": "Bolleria",
                "subsubcategory": "",
                "price_unit": 0.55,
                "unit_size": 0.05,
                "size_format": "kg",
            },
            {
                "product_id": 20727,
                "product_name": "Mantequilla con sal Hacendado",
                "category": "Huevos, leche y mantequilla",
                "subcategory": "Mantequilla y margarina",
                "subsubcategory": "",
                "price_unit": 2.45,
                "unit_size": 0.25,
                "size_format": "kg",
            },
            {
                "product_id": 29034,
                "product_name": "Refresco cola Hacendado Zero azúcar",
                "category": "Agua y refrescos",
                "subcategory": "Refrescos",
                "subsubcategory": "",
                "price_unit": 0.35,
                "unit_size": 0.33,
                "size_format": "l",
            },
            {
                "product_id": 19897,
                "product_name": "Azúcar blanco Hacendado",
                "category": "Azucar, caramelos y chocolate",
                "subcategory": "Azucar",
                "subsubcategory": "",
                "price_unit": 1.0,
                "unit_size": 1.0,
                "size_format": "kg",
            },
            {
                "product_id": 31540,
                "product_name": "Huevos grandes L",
                "category": "Huevos, leche y mantequilla",
                "subcategory": "Huevos",
                "subsubcategory": "",
                "price_unit": 1.8,
                "unit_size": 6.0,
                "size_format": "ud",
            },
            {
                "product_id": 83284,
                "product_name": "Panecillo harina integral 50% de trigo sin sal añadida",
                "category": "Panaderia y pasteleria",
                "subcategory": "Pan",
                "subsubcategory": "",
                "price_unit": 0.33,
                "unit_size": 0.08,
                "size_format": "kg",
            },
            {
                "product_id": 29100,
                "product_name": "Harina de trigo Hacendado",
                "category": "Arroz, legumbres y pasta",
                "subcategory": "Harina y pan rallado",
                "subsubcategory": "",
                "price_unit": 0.72,
                "unit_size": 1.0,
                "size_format": "kg",
            },
            {
                "product_id": 19729,
                "product_name": "Sal fina Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Sal y especias",
                "subsubcategory": "",
                "price_unit": 0.4,
                "unit_size": 1.0,
                "size_format": "kg",
            },
        ]
    )
    monkeypatch.setattr(mcp_app, "_load_catalog", lambda: fake_catalog)

    result = {
        "answer": "Ingredientes de Mercadona:\n\nNo se encontraron ingredientes/productos en Mercadona para esta receta.",
        "block_1": "No se encontraron ingredientes/productos en Mercadona para esta receta.",
        "block_3": (
            "Derrite 200 g de chocolate negro con 150 g de mantequilla. "
            "Bate 3 huevos con 150 g de azucar. Añade 100 g de harina y "
            "una pizca de sal."
        ),
        "cost_plan": [],
        "cost_summary": {
            "servings": 4,
            "total_purchase_eur": 0.0,
            "total_escandallo_eur": 0.0,
        },
        "inferred_ingredients": [],
    }

    payload = mcp_app._build_query_recipe_payload(
        pregunta="tarta de chocolate",
        result=result,
    )

    assert (
        "Chocolate negro fundir Hacendado"
        in payload.productos_mercadona_exactos
    )
    assert (
        "Mantequilla con sal Hacendado" in payload.productos_mercadona_exactos
    )
    assert "Azúcar blanco Hacendado" in payload.productos_mercadona_exactos
    assert "Harina de trigo Hacendado" in payload.productos_mercadona_exactos
    assert (
        "Croissant de mantequilla 26%"
        not in payload.productos_mercadona_exactos
    )
    assert (
        "Refresco cola Hacendado Zero azúcar"
        not in payload.productos_mercadona_exactos
    )
    assert (
        "Panecillo harina integral 50% de trigo sin sal añadida"
        not in payload.productos_mercadona_exactos
    )
    assert payload.ingredientes_mercadona_texto_literal.startswith(
        "- 1 ud Chocolate negro fundir Hacendado"
    )
    assert payload.coste_total_compra_eur == 8.72


def test_search_tool_result_wraps_mcp_json_text():
    result = mcp_app._build_search_tool_result(
        [
            {
                "id": "12531",
                "title": "Chocolate negro fundir Hacendado",
                "url": "recetona://producto/12531",
            }
        ]
    )

    assert result.structuredContent["results"][0]["id"] == "12531"
    wrapped_payload = json.loads(result.content[0].text)
    assert wrapped_payload["results"][0]["title"] == (
        "Chocolate negro fundir Hacendado"
    )


def test_fetch_tool_result_wraps_mcp_json_text():
    payload = {
        "id": "12531",
        "title": "Chocolate negro fundir Hacendado",
        "text": "Producto: Chocolate negro fundir Hacendado",
        "url": "recetona://producto/12531",
        "metadata": {},
    }

    result = mcp_app._build_fetch_tool_result(payload)

    assert result.structuredContent == payload
    assert json.loads(result.content[0].text)["id"] == "12531"


def test_recipe_widget_resource_meta_adds_optional_domain(monkeypatch):
    monkeypatch.setenv("RECETONA_WIDGET_DOMAIN", "https://api.wensicm.com")

    meta = mcp_app._recipe_widget_resource_meta()

    assert meta["ui"]["domain"] == "https://api.wensicm.com"
    assert meta["openai/widgetDomain"] == "https://api.wensicm.com"
