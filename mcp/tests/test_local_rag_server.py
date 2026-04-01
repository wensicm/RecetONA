from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import local_rag_server


def test_ensure_runtime_rag_cache_downloads_missing_files(
    monkeypatch, tmp_path: Path
):
    cache_dir = tmp_path / "runtime_cache"
    monkeypatch.setenv("RECETONA_RAG_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RECETONA_RAG_CACHE_S3_BUCKET", "bucket-prueba")
    monkeypatch.setenv("RECETONA_RAG_CACHE_S3_PREFIX", "recetona/rag_cache")

    downloaded_keys: list[tuple[str, str]] = []

    class FakeS3Client:
        def download_file(
            self, bucket_name: str, object_key: str, target: str
        ):
            downloaded_keys.append((bucket_name, object_key))
            Path(target).write_bytes(b"contenido")

    fake_boto3 = types.SimpleNamespace(
        client=lambda service_name, region_name=None: FakeS3Client()
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    status = local_rag_server.ensure_runtime_rag_cache(
        raise_on_error=True,
        eager_download=True,
    )

    assert status["missing_files"] == []
    assert (cache_dir / "chunks.csv").exists()
    assert (cache_dir / "embeddings.npy").exists()
    assert downloaded_keys == [
        ("bucket-prueba", "recetona/rag_cache/chunks.csv"),
        ("bucket-prueba", "recetona/rag_cache/embeddings.npy"),
    ]


def test_build_notebook_runtime_overrides_chat_model(
    monkeypatch, tmp_path: Path
):
    runtime_cells = [
        (
            "from pathlib import Path\n"
            "EMBED_MODEL = 'text-embedding-3-small'\n"
            "CHAT_MODEL = 'gpt-4.1-mini'\n"
            "BASE_DIR = Path('.')\n"
            "CACHE_DIR = BASE_DIR / 'rag_cache'\n"
        ),
        "def _clean(v):\n    return v\n",
        "client = object()\n",
        "def ensure_embeddings(chunks):\n    return []\nchunks = []\nembeddings = ensure_embeddings(chunks)\n",
        (
            "def ask_agent(message, **kwargs):\n"
            "    return {'answer': CHAT_MODEL, 'block_1': '', 'block_2': '', 'block_3': ''}\n"
        ),
    ]
    cache_dir = tmp_path / "runtime_cache"
    cache_dir.mkdir()

    monkeypatch.setenv("RECETONA_CHAT_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("RECETONA_EMBED_MODEL", "text-embedding-3-large")
    monkeypatch.setattr(
        local_rag_server,
        "ensure_runtime_rag_cache",
        lambda **kwargs: {
            "cache_dir": cache_dir,
            "missing_files": [],
        },
    )
    monkeypatch.setattr(
        local_rag_server,
        "_extract_code_cells",
        lambda notebook_path: runtime_cells,
    )
    monkeypatch.setattr(
        local_rag_server,
        "_select_runtime_cells",
        lambda code_cells: runtime_cells,
    )

    runtime = local_rag_server.build_notebook_runtime(
        tmp_path / "fake_notebook.ipynb"
    )

    assert runtime["EMBED_MODEL"] == "text-embedding-3-large"
    assert runtime["CHAT_MODEL"] == "gpt-5.4-nano"


def test_notebook_rag_service_returns_cost_plan_and_summary():
    service = local_rag_server.NotebookRagService.__new__(
        local_rag_server.NotebookRagService
    )
    service._lock = threading.Lock()
    service._ask_agent = lambda message, **kwargs: {
        "answer": "respuesta final",
        "block_1": "- 1 ud Chocolate puro Valor (5,25€)",
        "block_2": "",
        "block_3": "Mezcla y hornea.",
        "cost_plan": pd.DataFrame(
            [
                {
                    "ingredient": "chocolate",
                    "product_name": "Chocolate puro Valor",
                }
            ]
        ),
        "cost_summary": {"total_purchase_eur": 5.25},
        "inferred_ingredients": ["chocolate"],
    }

    result = service.ask("tarta de chocolate")

    assert result["answer"] == "respuesta final"
    assert list(result["cost_plan"]["product_name"]) == [
        "Chocolate puro Valor"
    ]
    assert result["cost_summary"]["total_purchase_eur"] == 5.25
    assert result["inferred_ingredients"] == ["chocolate"]


def test_filter_incompatible_ingredient_candidates_drops_pastry_for_nuts():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "nueces",
                "product_name": "Trenza con nueces 4%",
                "category": "Bolleria",
                "subcategory": "Bolleria empaquetada",
            },
            {
                "ingredient": "nueces",
                "product_name": (
                    "Bifidus desnatado probioticos con nueces y cereales"
                ),
                "category": "Postres y yogures",
                "subcategory": "Yogures",
            },
            {
                "ingredient": "nueces",
                "product_name": "Nueces peladas",
                "category": "Frutos secos",
                "subcategory": "Naturales",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == ["Nueces peladas"]


def test_filter_incompatible_ingredient_candidates_drops_wrong_produce():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "zanahoria",
                "product_name": "Pepino",
                "category": "Fruta y verdura",
                "subcategory": "Verdura",
                "subsubcategory": "Pepino y zanahoria",
            },
            {
                "ingredient": "zanahoria",
                "product_name": "Zanahorias",
                "category": "Fruta y verdura",
                "subcategory": "Verdura",
                "subsubcategory": "Pepino y zanahoria",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == ["Zanahorias"]


def test_filter_incompatible_ingredient_candidates_drops_bread_for_garlic():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "ajo",
                "product_name": "Picatostes fritos con ajo Hacendado",
                "category": "Panadería y pastelería",
                "subcategory": "Picos, rosquilletas y picatostes",
                "subsubcategory": "Picatostes",
            },
            {
                "ingredient": "ajo",
                "product_name": "Ajo troceado Hacendado ultracongelado",
                "category": "Congelados",
                "subcategory": "Verdura",
                "subsubcategory": "Verdura",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == [
        "Ajo troceado Hacendado ultracongelado"
    ]


def test_filter_incompatible_ingredient_candidates_drops_wrong_broth():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "caldo de carne",
                "product_name": "Caldo de pollo Hacendado",
                "category": "Conservas, caldos y cremas",
                "subcategory": "Sopa y caldo",
                "subsubcategory": "Caldo líquido",
            },
            {
                "ingredient": "caldo de carne",
                "product_name": "Caldo de carne con sofrito Hacendado",
                "category": "Conservas, caldos y cremas",
                "subcategory": "Sopa y caldo",
                "subsubcategory": "Caldo líquido",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == [
        "Caldo de carne con sofrito Hacendado"
    ]


def test_filter_incompatible_ingredient_candidates_drops_colorant_for_paprika():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "pimentón",
                "product_name": "Colorante alimentario Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Especias",
                "subsubcategory": "Colorante y pimentón",
            },
            {
                "ingredient": "pimentón",
                "product_name": "Pimentón dulce Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Especias",
                "subsubcategory": "Colorante y pimentón",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == ["Pimentón dulce Hacendado"]


def test_filter_incompatible_ingredient_candidates_uses_recipe_query_for_stew():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "carne de res",
                "product_name": "Botifarrón de carne",
                "category": "Carne",
                "subcategory": "Embutido",
                "subsubcategory": "Embutido",
            },
            {
                "ingredient": "carne de res",
                "product_name": "Preparado de carne picada vacuno",
                "category": "Carne",
                "subcategory": "Hamburguesas y picadas",
                "subsubcategory": "Picadas y otros",
            },
            {
                "ingredient": "carne de res",
                "product_name": "Trozo de vacuno para cocido",
                "category": "Carne",
                "subcategory": "Arreglos",
                "subsubcategory": "Arreglos",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits,
        recipe_query="receta de estofado de carne",
    )

    assert list(filtered["product_name"]) == ["Trozo de vacuno para cocido"]


def test_filter_incompatible_ingredient_candidates_keeps_original_if_empty():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "nueces",
                "product_name": "Trenza con nueces 4%",
                "category": "Bolleria",
                "subcategory": "Bolleria empaquetada",
            }
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == ["Trenza con nueces 4%"]


def test_filter_incompatible_ingredient_candidates_drops_pet_food_for_gelatin():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "gelatina",
                "product_name": (
                    "Trozos en gelatina gato adulto esterilizado "
                    "Delikuit con buey"
                ),
                "category": "Mascotas",
                "subcategory": "Gato",
                "subsubcategory": "Alimentación húmeda",
            },
            {
                "ingredient": "gelatina",
                "product_name": "Gelatina neutra en hojas Hacendado",
                "category": "Postres y yogures",
                "subcategory": "Repostería",
                "subsubcategory": "Gelatina y decoración",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == [
        "Gelatina neutra en hojas Hacendado"
    ]


def test_filter_incompatible_ingredient_candidates_returns_empty_for_only_non_food():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "gelatina",
                "product_name": (
                    "Trozos en gelatina gato adulto esterilizado "
                    "Delikuit con buey"
                ),
                "category": "Mascotas",
                "subcategory": "Gato",
                "subsubcategory": "Alimentación húmeda",
            }
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert filtered.empty


def test_filter_incompatible_ingredient_candidates_drops_laurel_for_gelatin():
    df_hits = pd.DataFrame(
        [
            {
                "ingredient": "gelatina en hojas",
                "product_name": "Hoja de laurel Hacendado",
                "category": "Aceite, especias y salsas",
                "subcategory": "Especias",
                "subsubcategory": "Hierbas aromáticas",
            },
            {
                "ingredient": "gelatina en hojas",
                "product_name": "Gelatina sabor fresa Hacendado",
                "category": "Postres y yogures",
                "subcategory": "Gelatina y otros postres",
                "subsubcategory": "Gelatina",
            },
        ]
    )

    filtered = local_rag_server._filter_incompatible_ingredient_candidates(
        df_hits
    )

    assert list(filtered["product_name"]) == ["Gelatina sabor fresa Hacendado"]


def test_collect_missing_recipe_ingredients_ignores_available_and_duplicates():
    plan_df = pd.DataFrame(
        [
            {
                "ingredient": "gelatina neutra",
                "product_name": None,
            },
            {
                "ingredient": "gelatina neutra",
                "product_name": None,
            },
            {
                "ingredient": "chocolate negro",
                "product_name": "Chocolate negro fundir Hacendado",
            },
            {
                "ingredient": "vainilla",
                "product_name": "",
            },
        ]
    )

    missing = local_rag_server._collect_missing_recipe_ingredients(plan_df)

    assert missing == ["gelatina neutra", "vainilla"]


def test_build_recipe_generation_prompt_forces_available_ingredients_only():
    plan_df = pd.DataFrame(
        [
            {
                "ingredient": "chocolate negro",
                "product_name": "Chocolate negro fundir Hacendado",
            },
            {
                "ingredient": "gelatina neutra",
                "product_name": None,
            },
        ]
    )

    prompt = local_rag_server._build_recipe_generation_prompt(
        "Receta de tarta de tres chocolates",
        plan_df,
        max_chars=1000,
    )

    assert "Chocolate negro fundir Hacendado" in prompt
    assert "Ingredientes solicitados sin producto exacto disponible" in prompt
    assert "gelatina neutra" in prompt
    assert "Usa solo los productos disponibles listados arriba." in prompt
    assert "adapta la receta a lo disponible" in prompt
    assert "No menciones ingredientes que no aparezcan" in prompt
    assert (
        "Respeta el formato y tipo literal del producto disponible" in prompt
    )
