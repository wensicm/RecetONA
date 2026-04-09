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


def test_build_query_recipe_payload_returns_widget_model(monkeypatch):
    fake_catalog = pd.DataFrame(
        [
            {
                "product_id": 12531,
                "product_name": "Chocolate negro fundir Hacendado",
                "unit_size": 0.2,
                "size_format": "kg",
                "nutrition_kcal_100": 529.0,
                "thumbnail_url": (
                    "https://prod-mercadona.imgix.net/images/"
                    "chocolate-negro.jpg"
                ),
            }
        ]
    )
    monkeypatch.setattr(mcp_app, "_load_catalog", lambda: fake_catalog)
    result = {
        "answer": "respuesta literal completa",
        "block_1": "- 1 ud Chocolate negro fundir Hacendado (2,35€)",
        "block_3": (
            "Derrite 100 g de Chocolate negro fundir Hacendado y mezcla."
        ),
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
    assert payload.ingredientes_mercadona[0].imagen_url == (
        "https://prod-mercadona.imgix.net/images/chocolate-negro.jpg"
    )
    assert payload.calorias_totales_kcal == 529.0


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
    assert result.meta["recetona/widgetVersion"] == "recipe-v9"
    assert "panel adjunto" in result.content[0].text


def test_render_recipe_tool_meta_links_widget_template():
    meta = mcp_app._render_recipe_tool_meta()

    assert meta["ui"]["resourceUri"] == mcp_app.RECIPE_WIDGET_URI
    assert meta["openai/outputTemplate"] == mcp_app.RECIPE_WIDGET_URI


def test_query_recipe_data_tool_meta_links_widget_template():
    meta = mcp_app._query_recipe_data_tool_meta()

    assert meta["ui"]["resourceUri"] == mcp_app.RECIPE_WIDGET_URI
    assert meta["openai/outputTemplate"] == mcp_app.RECIPE_WIDGET_URI


def test_recipe_validation_reasoning_defaults_to_none(monkeypatch):
    monkeypatch.delenv(
        "RECETONA_RECIPE_VALIDATION_REASONING_EFFORT",
        raising=False,
    )
    monkeypatch.delenv("RECETONA_REASONING_EFFORT", raising=False)

    assert mcp_app._get_recipe_validation_reasoning() == {"effort": "none"}


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
                "nutrition_kcal_100": 529.0,
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
                "nutrition_kcal_100": 406.0,
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
                "nutrition_kcal_100": 742.0,
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
                "nutrition_kcal_100": 0.0,
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
                "nutrition_kcal_100": 400.0,
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
                "nutrition_kcal_100": 150.0,
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
                "nutrition_kcal_100": 250.0,
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
                "nutrition_kcal_100": 364.0,
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
                "nutrition_kcal_100": None,
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
    assert payload.calorias_totales_kcal is not None
    assert payload.calorias_totales_kcal > 0


def test_build_query_recipe_payload_adds_missing_ingredient_after_validation(
    monkeypatch,
):
    fake_catalog = pd.DataFrame(
        [
            {
                "product_id": 13209,
                "product_name": ("Cebolla troceada Hacendado ultracongelada"),
                "category": "Verduras",
                "subcategory": "Verduras ultracongeladas",
                "subsubcategory": "",
                "price_unit": 1.04,
                "unit_size": 0.45,
                "size_format": "kg",
                "nutrition_kcal_100": 30.0,
            },
            {
                "product_id": 56789,
                "product_name": "Filetes pechuga de pollo",
                "category": "Carne",
                "subcategory": "Aves y pollo",
                "subsubcategory": "Pollo",
                "price_unit": 4.5,
                "unit_size": 0.55,
                "size_format": "kg",
                "nutrition_kcal_100": 110.0,
            },
        ]
    )
    monkeypatch.setattr(mcp_app, "_load_catalog", lambda: fake_catalog)
    monkeypatch.setattr(
        mcp_app,
        "_validate_recipe_ingredients_with_agent",
        lambda **_: ["pollo"],
    )

    result = {
        "answer": "respuesta literal completa",
        "block_1": "- 1 ud Cebolla troceada Hacendado ultracongelada (1,04€)",
        "block_3": (
            "Sofrie cebolla. Agrega 500 g de pollo troceado y cocina "
            "hasta que se dore."
        ),
        "cost_plan": [
            {
                "ingredient": "cebolla",
                "required_qty": 1,
                "required_unit": "ud",
                "product_id": 13209,
                "product_name": ("Cebolla troceada Hacendado ultracongelada"),
                "unit_size": 0.45,
                "size_format": "kg",
                "price_unit": 1.04,
                "units_to_buy": 1,
                "purchase_cost_eur": 1.04,
                "escandallo_cost_eur": 0.3,
                "notes": "OK",
            }
        ],
        "cost_summary": {
            "servings": 4,
            "total_purchase_eur": 1.04,
            "total_escandallo_eur": 0.3,
        },
        "inferred_ingredients": ["cebolla"],
    }

    payload = mcp_app._build_query_recipe_payload(
        pregunta="pollo al curry",
        result=result,
    )

    assert "Filetes pechuga de pollo" in payload.productos_mercadona_exactos
    assert "pollo" in payload.ingredientes_inferidos
    assert "Filetes pechuga de pollo" in payload.respuesta_literal_mcp
    assert payload.coste_total_compra_eur == 5.54


def test_build_query_recipe_payload_replaces_implausible_product(
    monkeypatch,
):
    fake_catalog = pd.DataFrame(
        [
            {
                "product_id": 34125,
                "product_name": "Curry Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Especias",
                "subsubcategory": "Otras especias",
                "price_unit": 1.1,
                "unit_size": 0.06,
                "size_format": "kg",
                "nutrition_kcal_100": 325.0,
            },
            {
                "product_id": 60574,
                "product_name": "Pimentón picante Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Sal y especias",
                "subsubcategory": "",
                "price_unit": 1.3,
                "unit_size": 0.075,
                "size_format": "kg",
                "nutrition_kcal_100": 282.0,
            },
            {
                "product_id": 13209,
                "product_name": ("Cebolla troceada Hacendado ultracongelada"),
                "category": "Verduras",
                "subcategory": "Verduras ultracongeladas",
                "subsubcategory": "",
                "price_unit": 1.04,
                "unit_size": 0.45,
                "size_format": "kg",
                "nutrition_kcal_100": 30.0,
            },
        ]
    )
    monkeypatch.setattr(mcp_app, "_load_catalog", lambda: fake_catalog)
    monkeypatch.setattr(
        mcp_app,
        "_validate_recipe_ingredients_with_agent",
        lambda **_: [],
    )
    monkeypatch.setattr(
        mcp_app,
        "_validate_recipe_substitutions_with_agent",
        lambda **_: [
            mcp_app.RecipeIngredientReplacementSuggestion(
                current_product_name="Pimentón picante Hacendado",
                replacement_ingredient="curry",
            )
        ],
    )

    result = {
        "answer": "respuesta literal completa",
        "block_1": (
            "- 1 ud Cebolla troceada Hacendado ultracongelada (1,04€)\n"
            "- 1 ud Pimentón picante Hacendado (1,30€)"
        ),
        "block_3": (
            "Sofrie cebolla y añade curry para la salsa del pollo al curry."
        ),
        "cost_plan": [
            {
                "ingredient": "cebolla",
                "required_qty": 1,
                "required_unit": "ud",
                "product_id": 13209,
                "product_name": ("Cebolla troceada Hacendado ultracongelada"),
                "unit_size": 0.45,
                "size_format": "kg",
                "price_unit": 1.04,
                "units_to_buy": 1,
                "purchase_cost_eur": 1.04,
                "escandallo_cost_eur": 0.3,
                "notes": "OK",
            },
            {
                "ingredient": "curry en polvo",
                "required_qty": 1,
                "required_unit": "ud",
                "product_id": 60574,
                "product_name": "Pimentón picante Hacendado",
                "unit_size": 0.075,
                "size_format": "kg",
                "price_unit": 1.3,
                "units_to_buy": 1,
                "purchase_cost_eur": 1.3,
                "escandallo_cost_eur": 0.08,
                "notes": "OK",
            },
        ],
        "cost_summary": {
            "servings": 4,
            "total_purchase_eur": 2.34,
            "total_escandallo_eur": 0.38,
        },
        "inferred_ingredients": ["cebolla", "curry en polvo"],
    }

    payload = mcp_app._build_query_recipe_payload(
        pregunta="pollo al curry",
        result=result,
    )

    assert "Curry Hacendado" in payload.productos_mercadona_exactos
    assert (
        "Pimentón picante Hacendado" not in payload.productos_mercadona_exactos
    )
    assert "curry" in payload.ingredientes_inferidos
    assert "Curry Hacendado" in payload.respuesta_literal_mcp
    assert payload.coste_total_compra_eur == 2.14


def test_build_query_recipe_payload_adds_inferred_missing_ingredients(
    monkeypatch,
):
    fake_catalog = pd.DataFrame(
        [
            {
                "product_id": 13209,
                "product_name": ("Cebolla troceada Hacendado ultracongelada"),
                "category": "Verduras",
                "subcategory": "Verduras ultracongeladas",
                "subsubcategory": "",
                "price_unit": 1.04,
                "unit_size": 0.45,
                "size_format": "kg",
            },
            {
                "product_id": 34125,
                "product_name": "Curry Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Especias",
                "subsubcategory": "Otras especias",
                "price_unit": 1.1,
                "unit_size": 0.06,
                "size_format": "kg",
            },
            {
                "product_id": 32150,
                "product_name": "Caldo de pollo Hacendado",
                "category": "Caldos, sopas y cremas",
                "subcategory": "Caldos",
                "subsubcategory": "Pollo",
                "price_unit": 0.8,
                "unit_size": 1.0,
                "size_format": "l",
            },
            {
                "product_id": 60573,
                "product_name": "Pimentón dulce Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Sal y especias",
                "subsubcategory": "",
                "price_unit": 1.3,
                "unit_size": 0.075,
                "size_format": "kg",
            },
        ]
    )
    monkeypatch.setattr(mcp_app, "_load_catalog", lambda: fake_catalog)
    monkeypatch.setattr(
        mcp_app,
        "_validate_recipe_ingredients_with_agent",
        lambda **_: [],
    )
    monkeypatch.setattr(
        mcp_app,
        "_validate_recipe_substitutions_with_agent",
        lambda **_: [],
    )

    result = {
        "answer": "respuesta literal completa",
        "block_1": (
            "- 1 ud Cebolla troceada Hacendado ultracongelada (1,04€)\n"
            "- 1 ud Pimentón dulce Hacendado (1,30€)"
        ),
        "block_3": (
            "Sofrie cebolla. Añade pimenton. "
            "Vierte leche de coco y caldo de pollo. "
            "Termina el pollo al curry y sirve."
        ),
        "cost_plan": [
            {
                "ingredient": "cebolla",
                "required_qty": 1,
                "required_unit": "ud",
                "product_id": 13209,
                "product_name": ("Cebolla troceada Hacendado ultracongelada"),
                "unit_size": 0.45,
                "size_format": "kg",
                "price_unit": 1.04,
                "units_to_buy": 1,
                "purchase_cost_eur": 1.04,
                "escandallo_cost_eur": 0.3,
                "notes": "OK",
            },
            {
                "ingredient": "pimenton",
                "required_qty": 1,
                "required_unit": "ud",
                "product_id": 60573,
                "product_name": "Pimentón dulce Hacendado",
                "unit_size": 0.075,
                "size_format": "kg",
                "price_unit": 1.3,
                "units_to_buy": 1,
                "purchase_cost_eur": 1.3,
                "escandallo_cost_eur": 0.08,
                "notes": "OK",
            },
        ],
        "cost_summary": {
            "servings": 4,
            "total_purchase_eur": 2.34,
            "total_escandallo_eur": 0.38,
        },
        "inferred_ingredients": [
            "cebolla",
            "curry en polvo",
            "caldo de pollo",
        ],
    }

    payload = mcp_app._build_query_recipe_payload(
        pregunta="pollo al curry",
        result=result,
    )

    assert "Curry Hacendado" in payload.productos_mercadona_exactos
    assert "Caldo de pollo Hacendado" in payload.productos_mercadona_exactos
    assert payload.coste_total_compra_eur == 4.24


def test_rule_based_replacement_suggestions_detect_curry_mismatch(
    monkeypatch,
):
    fake_catalog = pd.DataFrame(
        [
            {
                "product_id": 34125,
                "product_name": "Curry Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Especias",
                "subsubcategory": "Otras especias",
                "price_unit": 1.1,
                "unit_size": 0.06,
                "size_format": "kg",
            },
            {
                "product_id": 60574,
                "product_name": "Pimentón picante Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Sal y especias",
                "subsubcategory": "",
                "price_unit": 1.3,
                "unit_size": 0.075,
                "size_format": "kg",
            },
        ]
    )
    monkeypatch.setattr(mcp_app, "_load_catalog", lambda: fake_catalog)

    suggestions = mcp_app._build_rule_based_replacement_suggestions(
        [
            {
                "ingrediente_objetivo": "curry en polvo",
                "producto_mercadona": "Pimentón picante Hacendado",
                "producto_id": "60574",
                "tamano_envase_valor": 0.075,
                "tamano_envase_unidad": "kg",
            }
        ]
    )

    assert suggestions == [
        mcp_app.RecipeIngredientReplacementSuggestion(
            current_product_name="Pimentón picante Hacendado",
            replacement_ingredient="curry",
        )
    ]


def test_row_compatibility_rejects_flavored_gelatin_for_generic_gelatin():
    row = pd.Series(
        {
            "product_name": "Gelatina sabor fresa Hacendado",
            "category": "Postres y yogures",
            "subcategory": "Gelatina y otros postres",
            "subsubcategory": "Gelatina",
        }
    )

    assert not mcp_app._row_is_compatible_for_fallback_ingredient(
        ingredient="gelatina",
        row=row,
    )


def test_row_compatibility_accepts_philadelphia_for_queso_crema():
    row = pd.Series(
        {
            "product_name": "Queso untar original Philadelphia",
            "category": "Charcutería y quesos",
            "subcategory": ("Queso untable, fresco y especialidades"),
            "subsubcategory": "Queso untable",
        }
    )

    assert mcp_app._row_is_compatible_for_fallback_ingredient(
        ingredient="queso crema",
        row=row,
    )


def test_row_compatibility_rejects_camembert_for_queso_crema():
    row = pd.Series(
        {
            "product_name": "Crema de queso camembert Hacendado",
            "category": "Charcutería y quesos",
            "subcategory": ("Queso untable, fresco y especialidades"),
            "subsubcategory": "Queso untable",
        }
    )

    assert not mcp_app._row_is_compatible_for_fallback_ingredient(
        ingredient="queso crema",
        row=row,
    )


def test_optional_recipe_context_filter_ignores_serving_suggestions():
    recipe_text = (
        "Cocina el pollo hasta que se dore. "
        "Agrega un poco de agua si es necesario. "
        "Sirve caliente acompañado con arroz o pan."
    )

    assert mcp_app._ingredient_appears_only_in_optional_recipe_context(
        ingredient="agua",
        recipe_text=recipe_text,
    )
    assert mcp_app._ingredient_appears_only_in_optional_recipe_context(
        ingredient="arroz",
        recipe_text=recipe_text,
    )
    assert mcp_app._ingredient_appears_only_in_optional_recipe_context(
        ingredient="pan",
        recipe_text=recipe_text,
    )
    assert not mcp_app._ingredient_appears_only_in_optional_recipe_context(
        ingredient="pollo",
        recipe_text=recipe_text,
    )


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

    assert meta["ui"]["csp"]["resourceDomains"] == [
        "https://prod-mercadona.imgix.net"
    ]
    assert meta["openai/widgetCSP"]["resource_domains"] == [
        "https://prod-mercadona.imgix.net"
    ]
    assert meta["ui"]["domain"] == "https://api.wensicm.com"
    assert meta["openai/widgetDomain"] == "https://api.wensicm.com"
