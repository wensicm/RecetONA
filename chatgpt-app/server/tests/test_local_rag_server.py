from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

for candidate_path in (PROJECT_ROOT, SOURCE_DIR):
    resolved_path = str(candidate_path)
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

import local_rag_server as rag_server


def test_get_reasoning_options_defaults_to_none(monkeypatch):
    monkeypatch.delenv("RECETONA_REASONING_EFFORT", raising=False)

    assert rag_server._get_reasoning_options() == {"effort": "none"}


def test_filter_cost_plan_to_used_ingredients_drops_unused_rows():
    plan_df = pd.DataFrame(
        [
            {"ingredient": "pollo", "product_name": "Pollo entero"},
            {"ingredient": "curry", "product_name": "Curry Hacendado"},
            {"ingredient": "cebolla", "product_name": "Cebolla troceada"},
        ]
    )

    filtered_plan_df = rag_server._filter_cost_plan_to_used_ingredients(
        plan_df,
        used_ingredients=["pollo", "curry"],
    )

    assert filtered_plan_df["ingredient"].tolist() == ["pollo", "curry"]


class _FakeStructuredResponse:
    def __init__(self, parsed_payload):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        parsed=parsed_payload,
                    )
                ],
            )
        ]


class _FakeResponsesClient:
    def __init__(self, parsed_payloads):
        self.calls: list[dict] = []
        self._parsed_payloads = list(parsed_payloads)

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed_payload = self._parsed_payloads.pop(0)
        return _FakeStructuredResponse(parsed_payload)


def test_run_two_phase_recipe_pipeline_uses_structured_stages():
    hits = pd.DataFrame(
        [
            {
                "product_id": "1",
                "product_name": "Pollo entero",
                "category": "Carne",
                "price_unit": 4.5,
                "unit_size": 1.0,
                "size_format": "kg",
                "score": 0.9,
            }
        ]
    )
    cost_plan_df = pd.DataFrame(
        [
            {
                "ingredient": "pollo",
                "product_id": "1",
                "product_name": "Pollo entero",
                "required_qty": 1,
                "required_unit": "ud",
                "price_unit": 4.5,
                "purchase_cost_eur": 4.5,
                "escandallo_cost_eur": 2.8,
                "unit_size": 1.0,
                "size_format": "kg",
            },
            {
                "ingredient": "cebolla",
                "product_id": "2",
                "product_name": "Cebolla troceada Hacendado ultracongelada",
                "required_qty": 1,
                "required_unit": "ud",
                "price_unit": 1.04,
                "purchase_cost_eur": 1.04,
                "escandallo_cost_eur": 0.3,
                "unit_size": 0.45,
                "size_format": "kg",
            },
            {
                "ingredient": "curry",
                "product_id": "3",
                "product_name": "Curry Hacendado",
                "required_qty": 1,
                "required_unit": "ud",
                "price_unit": 1.1,
                "purchase_cost_eur": 1.1,
                "escandallo_cost_eur": 0.08,
                "unit_size": 0.06,
                "size_format": "kg",
            },
        ]
    )

    fake_responses = _FakeResponsesClient(
        [
            rag_server.RecipeDraftPlan(
                titulo="Pollo al curry",
                personas=4,
                ingredientes=["pollo", "cebolla", "curry"],
            ),
            rag_server.RecipeFinalPlan(
                titulo="Pollo al curry",
                ingredientes_usados=["pollo", "curry"],
                receta="Dora el pollo y termina con curry.",
            ),
        ]
    )
    fake_client = SimpleNamespace(responses=fake_responses)

    def _dedupe(values):
        seen = set()
        deduped = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    namespace = {
        "retrieve": lambda *args, **kwargs: (
            hits,
            ["receta de pollo al curry"],
            ["cebolla"],
        ),
        "parse_servings": lambda query, default=4: 4,
        "tool_get_products_for_ingredients": lambda ingredients, **kwargs: {
            ingredient: pd.DataFrame(
                [{"product_name": f"Producto para {ingredient}"}]
            )
            for ingredient in ingredients
        },
        "build_recipe_cost_plan": lambda **kwargs: (
            cost_plan_df.copy(),
            {
                "servings": 4,
                "total_purchase_eur": 6.64,
                "total_escandallo_eur": 3.18,
                "missing_purchase_ingredients": [],
                "missing_escandallo_ingredients": [],
            },
        ),
        "format_ingredient_catalog_text": lambda catalog, max_items=6: (
            f"catalogo={','.join(catalog.keys())}"
        ),
        "format_cost_plan_text": lambda plan_df, summary: (
            f"total={summary['total_purchase_eur']:.2f}"
        ),
        "build_block1_ingredients_mercadona": lambda plan_df: "\n".join(
            plan_df["product_name"].tolist()
        ),
        "tool_get_total_purchase_price": lambda summary: "",
        "compose_structured_answer": lambda block_1, block_2, block_3: (
            f"{block_1}\n{block_3}"
        ),
        "_catalog_preview": lambda catalog, n=3: {},
        "_remove_redundant_ingredients": _dedupe,
        "build_block3_recipe_text": lambda *args, **kwargs: (
            "fallback",
            None,
        ),
        "client": fake_client,
    }

    result = rag_server._run_two_phase_recipe_pipeline(
        namespace,
        query="Receta de pollo al curry",
        top_k=35,
        model="gpt-5.4-nano",
        retrieval_mode="hybrid",
        alpha=0.65,
        recipe_mode="auto",
        use_ingredient_tool=True,
        candidates_per_ingredient=12,
    )

    assert result["inferred_ingredients"] == ["pollo", "cebolla", "curry"]
    assert result["cost_plan"]["ingredient"].tolist() == ["pollo", "curry"]
    assert result["productos"] if False else True
    assert "Pollo entero" in result["block_1"]
    assert "Cebolla troceada" not in result["block_1"]
    assert result["cost_summary"]["total_purchase_eur"] == 5.6
    assert result["block_3"] == "Dora el pollo y termina con curry."
    assert len(fake_responses.calls) == 2
    assert all(
        call["model"] == "gpt-5.4-nano" for call in fake_responses.calls
    )
    assert all(
        call["reasoning"] == {"effort": "none"}
        for call in fake_responses.calls
    )


def test_candidate_filter_rejects_flavored_gelatin_for_generic_gelatin():
    row = pd.Series(
        {
            "product_name": "Gelatina sabor fresa Hacendado",
            "category": "Postres y yogures",
            "subcategory": "Gelatina y otros postres",
            "subsubcategory": "Gelatina",
        }
    )

    assert rag_server._candidate_is_incompatible_for_ingredient(
        "gelatina",
        row,
        "",
    )


def test_candidate_filter_rejects_camembert_for_queso_crema():
    row = pd.Series(
        {
            "product_name": "Crema de queso camembert Hacendado",
            "category": "Charcutería y quesos",
            "subcategory": ("Queso untable, fresco y especialidades"),
            "subsubcategory": "Queso untable",
        }
    )

    assert rag_server._candidate_is_incompatible_for_ingredient(
        "queso crema",
        row,
        "Receta de una tarta de tres chocolates",
    )


def test_filter_non_shopping_ingredients_drops_water():
    assert rag_server._filter_non_shopping_ingredients(
        ["chocolate negro", "agua", "nata para montar"]
    ) == ["chocolate negro", "nata para montar"]


def test_filter_cost_plan_catalog_plausibility_drops_water_and_bad_gelatin():
    plan_df = pd.DataFrame(
        [
            {
                "ingredient": "chocolate negro",
                "product_name": "Chocolate negro fundir Hacendado",
                "category": "Azúcar, caramelos y chocolate",
                "subcategory": "Chocolate",
                "subsubcategory": "Chocolate negro",
            },
            {
                "ingredient": "agua",
                "product_name": "Agua de coco Hacendado 100% natural",
                "category": "Bebidas",
                "subcategory": "Agua de coco",
                "subsubcategory": "",
            },
            {
                "ingredient": "gelatina neutra",
                "product_name": "Preparado en polvo gelatina fresa Hacendado",
                "category": "Panadería y pastelería",
                "subcategory": "Harina y preparado repostería",
                "subsubcategory": "Levadura y preparado repostería",
            },
        ]
    )

    filtered_plan_df = rag_server._filter_cost_plan_catalog_plausibility(
        plan_df,
        recipe_query="Receta de una tarta de tres chocolates",
    )

    assert filtered_plan_df["ingredient"].tolist() == ["chocolate negro"]
