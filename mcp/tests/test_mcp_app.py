from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import recetona.mcp_app as mcp_app
from recetona.mcp_app import build_streamable_http_app


def test_public_health_route_is_not_exposed():
    client = TestClient(build_streamable_http_app())

    response = client.get("/health")

    assert response.status_code == 404


def test_mcp_options_preflight_exposes_cors_headers():
    client = TestClient(build_streamable_http_app())

    response = client.options(
        "/",
        headers={
            "Origin": "https://chatgpt.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_recipe_runtime_status_marks_ssm_origin(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_SSM_PARAMETER", "OPENAI_API_KEY")
    monkeypatch.setattr(mcp_app, "_resolved_openai_api_key", None)
    monkeypatch.setattr(mcp_app, "_resolved_openai_api_key_source", None)

    fake_ssm_client = types.SimpleNamespace(
        get_parameter=lambda **_kwargs: {
            "Parameter": {"Value": "ssm-secret-value"}
        }
    )
    fake_boto3 = types.SimpleNamespace(
        client=lambda service_name, region_name=None: fake_ssm_client
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    status = mcp_app._recipe_runtime_status()

    assert status["openai_api_key_present"] is True
    assert status["openai_api_key_source"] == "ssm"
    assert status["openai_api_key_ssm_parameter"] == "OPENAI_API_KEY"


def test_build_query_recipe_payload_preserves_exact_product_names():
    result = {
        "answer": "respuesta literal completa",
        "block_1": "- 1 ud Chocolate puro Valor (5,25€)",
        "block_3": "Mezcla y hornea.",
        "cost_plan": pd.DataFrame(
            [
                {
                    "ingredient": "chocolate",
                    "required_qty": 1,
                    "required_unit": "ud",
                    "product_id": 1234,
                    "product_name": "Chocolate puro Valor",
                    "unit_size": 250,
                    "size_format": "g",
                    "price_unit": 5.25,
                    "units_to_buy": 1,
                    "purchase_cost_eur": 5.25,
                    "escandallo_cost_eur": 5.25,
                    "notes": "OK",
                }
            ]
        ),
        "cost_summary": {
            "servings": 4,
            "total_purchase_eur": 5.25,
            "total_escandallo_eur": 5.25,
        },
        "inferred_ingredients": ["chocolate"],
    }

    payload = mcp_app._build_query_recipe_payload(
        pregunta="tarta de chocolate",
        result=result,
    )

    assert payload["productos_mercadona_exactos"] == ["Chocolate puro Valor"]
    assert payload["ingredientes_mercadona"][0]["producto_mercadona"] == (
        "Chocolate puro Valor"
    )
    assert payload["ingredientes_mercadona_texto_literal"] == (
        "- 1 ud Chocolate puro Valor (5,25€)"
    )
    assert payload["coste_total_compra_eur"] == 5.25


def test_query_recipe_tool_result_keeps_structured_payload():
    payload = {
        "pregunta": "tarta de chocolate",
        "productos_mercadona_exactos": ["Chocolate puro Valor"],
        "ingredientes_mercadona": [],
    }

    result = mcp_app._build_query_recipe_tool_result(payload)

    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "panel adjunto" in result.content[0].text
    assert result.structuredContent == payload
    assert result.meta["recetona/widgetVersion"] == "recipe-v1"


def test_recipe_widget_html_uses_apps_bridge_and_tool_output():
    html = mcp_app._build_recipe_widget_html()

    assert "window.openai?.toolOutput" in html
    assert "ui/notifications/tool-result" in html
    assert "openai:set_globals" in html


def test_query_recipe_tool_meta_links_widget_template():
    meta = mcp_app._query_recipe_tool_meta()

    assert meta["ui"]["resourceUri"] == mcp_app.RECIPE_WIDGET_URI
    assert meta["openai/outputTemplate"] == mcp_app.RECIPE_WIDGET_URI
