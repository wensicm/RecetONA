"""
Mercadona product scraper
=========================

This script scrapes the public Mercadona API to build a structured
dataset of categories, sub‑categories and all available products. It
aggregates pricing information, packaging/quantity details and the
nutritional/ingredient labelling for each item.  The resulting data is
exported to an Excel workbook which can then be used for analysis
(price per unit comparisons, ingredient searches, etc.).

Requirements
------------

* Python 3.8 or newer
* `requests` (``pip install requests``)
* `pandas` (``pip install pandas openpyxl``)

Usage
-----

Running this script will download all categories, traverse each
sub‑category and product, and then save the data to an Excel file
called ``mercadona_data.xlsx`` in the current working directory.  If
you only want to scrape a particular set of categories or a single
postcode, you can modify the ``BASE_URL`` or filter logic accordingly.
The Mercadona API is public and does not require authentication, but
please use it responsibly.

Example::

    python mercadona_scraper_script.py

"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import re
import unicodedata
from urllib.parse import parse_qs, urlparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import pandas as pd
import requests
try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

# Base URL for Mercadona API.  You can prepend a language or postal
# code query string if necessary, but the default works across Spain.
BASE_URL = "https://tienda.mercadona.es/api"
LOG_EVERY_PRODUCTS = 100
LOG_EVERY_DOWNLOADS = 250
MAX_IMAGE_DOWNLOAD_WORKERS = 20
MAX_IMAGES_PER_ROW = 2
NUTRITION_COLUMNS = [
    "nutrition_kj_100",
    "nutrition_kcal_100",
    "nutrition_fat_g_100",
    "nutrition_saturates_g_100",
    "nutrition_carbs_g_100",
    "nutrition_sugars_g_100",
    "nutrition_protein_g_100",
    "nutrition_salt_g_100",
    "nutrition_ocr_text",
]
LOGGER = logging.getLogger(__name__)

@dataclass
class ProductInfo:
    category: str
    subcategory: str
    subsubcategory: Optional[str]
    product_id: str
    product_name: str
    packaging: Optional[str]
    unit_size: Optional[float]
    size_format: Optional[str]
    price_unit: Optional[float]
    price_bulk: Optional[float]
    thumbnail_url: Optional[str]
    photo_urls: Optional[str]
    ingredients: Optional[str]
    allergens: Optional[str]


def extract_image_urls(product_data: Dict) -> tuple[Optional[str], Optional[str]]:
    """Extract thumbnail and unique image URLs from a product payload."""
    original_thumbnail_url = product_data.get("thumbnail")
    photos = product_data.get("photos", []) or []
    photo_urls_list: List[str] = []
    seen = set()

    def image_size_score(url: Optional[str]) -> int:
        """Return a comparable score using query params h and w (area)."""
        if not isinstance(url, str) or not url:
            return -1
        try:
            query = parse_qs(urlparse(url).query)
            h = int(query.get("h", ["0"])[0])
            w = int(query.get("w", ["0"])[0])
            if h > 0 and w > 0:
                return h * w
        except (TypeError, ValueError):
            pass
        return 0

    for photo in photos:
        if isinstance(photo, dict):
            # Keep only the highest-resolution url for each photo entry.
            candidates = [photo.get(key) for key in ("thumbnail", "regular", "zoom", "url")]
            best_url = None
            best_score = -1
            for candidate in candidates:
                score = image_size_score(candidate)
                if score > best_score:
                    best_score = score
                    best_url = candidate
            if isinstance(best_url, str) and best_url and best_url not in seen:
                seen.add(best_url)
                photo_urls_list.append(best_url)
        elif isinstance(photo, str) and photo and photo not in seen:
            seen.add(photo)
            photo_urls_list.append(photo)
    photo_urls = " | ".join(photo_urls_list) if photo_urls_list else None
    thumbnail_url = photo_urls_list[0] if photo_urls_list else original_thumbnail_url
    return thumbnail_url, photo_urls


def normalize_product_id(raw_value) -> Optional[str]:
    """Normalize product id values from Excel into API-ready strings."""
    if pd.isna(raw_value):
        return None
    if isinstance(raw_value, float) and raw_value.is_integer():
        return str(int(raw_value))
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return None
    return value


def split_photo_urls(raw_value) -> List[str]:
    """Split photo_urls cell value into individual URLs."""
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [url.strip() for url in re.split(r"\s*\|\s*|\s*\n+\s*", text) if url.strip()]


def normalize_text(text: str) -> str:
    """Lowercase text without accents for easier matching."""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalized.lower()


def to_float(number_text: str) -> Optional[float]:
    """Parse decimal numbers with comma or dot."""
    if not isinstance(number_text, str):
        return None
    text = number_text.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def extract_energy_values(joined_text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract kJ and kcal values from OCR text."""
    kj_match = re.search(r"(\d{1,4}(?:[.,]\d+)?)\s*kj\b", joined_text, flags=re.IGNORECASE)
    kcal_match = re.search(r"(\d{1,4}(?:[.,]\d+)?)\s*kcal\b", joined_text, flags=re.IGNORECASE)
    kj_value = to_float(kj_match.group(1)) if kj_match else None
    kcal_value = to_float(kcal_match.group(1)) if kcal_match else None
    return kj_value, kcal_value


def extract_g_value_from_line(line: str) -> Optional[float]:
    """Extract first g/mg value from a line; mg is converted to grams."""
    for number_text, unit in re.findall(r"(\d{1,4}(?:[.,]\d+)?)\s*(mg|g)\b", line, flags=re.IGNORECASE):
        value = to_float(number_text)
        if value is None:
            continue
        if unit.lower() == "mg":
            return value / 1000.0
        return value
    # Fallback: first plausible numeric token
    for number_text in re.findall(r"(?<!\d)(\d{1,3}(?:[.,]\d+)?)", line):
        value = to_float(number_text)
        if value is not None and 0 <= value <= 100:
            return value
    return None


def find_keyword_index(lines_norm: List[str], keywords: List[str]) -> Optional[int]:
    """Find first line index containing any keyword."""
    for idx, line in enumerate(lines_norm):
        if any(keyword in line for keyword in keywords):
            return idx
    return None


def collect_values_after(
    lines_original: List[str],
    start_idx: Optional[int],
    stop_idx: Optional[int],
    max_lookahead: int = 7,
) -> List[float]:
    """Collect likely nutrient values after a keyword line."""
    if start_idx is None:
        return []
    values: List[float] = []
    end_idx = min(len(lines_original), start_idx + 1 + max_lookahead)
    if stop_idx is not None:
        end_idx = min(end_idx, stop_idx)
    for idx in range(start_idx, end_idx):
        value = extract_g_value_from_line(lines_original[idx])
        if value is None:
            continue
        # Avoid duplicates from OCR overlaps
        if not values or abs(values[-1] - value) > 1e-6:
            values.append(value)
    return values


def parse_nutrition_from_ocr_lines(lines_original: List[str]) -> Dict[str, Optional[float]]:
    """Parse core nutrition metrics from OCR lines."""
    joined_text = " | ".join(lines_original)
    lines_norm = [normalize_text(line) for line in lines_original]

    kj_value, kcal_value = extract_energy_values(joined_text)
    idx_fat = find_keyword_index(lines_norm, ["grasas", "lipidos", "fat"])
    idx_sat = find_keyword_index(lines_norm, ["saturad", "saturat"])
    idx_carbs = find_keyword_index(lines_norm, ["hidratos", "carbohidr", "carbohyd"])
    idx_sugars = find_keyword_index(lines_norm, ["azucar", "sugar"])
    idx_protein = find_keyword_index(lines_norm, ["proteina", "protein"])
    idx_salt = find_keyword_index(lines_norm, [" sal", "salt", "sal/"])

    fat_values = collect_values_after(lines_original, idx_fat, idx_carbs)
    carbs_values = collect_values_after(lines_original, idx_carbs, idx_protein)
    protein_values = collect_values_after(lines_original, idx_protein, idx_salt)
    salt_values = collect_values_after(lines_original, idx_salt, None)

    fat_value = fat_values[0] if fat_values else None
    sat_value = None
    if idx_sat is not None:
        sat_candidates = collect_values_after(lines_original, idx_sat, idx_carbs)
        sat_value = sat_candidates[0] if sat_candidates else None
    if sat_value is None and len(fat_values) > 1:
        sat_value = fat_values[1]

    carbs_value = carbs_values[0] if carbs_values else None
    sugar_value = None
    if idx_sugars is not None:
        sugar_candidates = collect_values_after(lines_original, idx_sugars, idx_protein)
        sugar_value = sugar_candidates[0] if sugar_candidates else None
    if sugar_value is None and len(carbs_values) > 1:
        sugar_value = carbs_values[1]

    protein_value = protein_values[0] if protein_values else None
    salt_value = salt_values[0] if salt_values else None

    return {
        "nutrition_kj_100": kj_value,
        "nutrition_kcal_100": kcal_value,
        "nutrition_fat_g_100": fat_value,
        "nutrition_saturates_g_100": sat_value,
        "nutrition_carbs_g_100": carbs_value,
        "nutrition_sugars_g_100": sugar_value,
        "nutrition_protein_g_100": protein_value,
        "nutrition_salt_g_100": salt_value,
        "nutrition_ocr_text": joined_text,
    }


def extract_nutrition_to_excel(
    excel_path: str,
    max_rows: Optional[int] = None,
) -> None:
    """Run OCR on nutrition_image_file and write parsed nutrition columns into Excel."""
    if RapidOCR is None:
        raise RuntimeError(
            "rapidocr_onnxruntime no esta disponible. Instala dependencias en la carpeta lib."
        )
    LOGGER.info("Extrayendo valor nutricional por OCR desde %s", excel_path)
    df = pd.read_excel(excel_path)
    if "nutrition_image_file" not in df.columns:
        LOGGER.warning("No existe columna 'nutrition_image_file'. Se omite OCR nutricional.")
        return

    for column in NUTRITION_COLUMNS:
        if column not in df.columns:
            df[column] = None

    engine = RapidOCR()
    processed = 0
    total_rows = len(df) if max_rows is None else min(len(df), max_rows)
    for row_index in range(total_rows):
        image_path = df.at[row_index, "nutrition_image_file"]
        if pd.isna(image_path):
            continue
        image_path = str(image_path).strip()
        if not image_path or not os.path.exists(image_path):
            continue

        try:
            ocr_result, _ = engine(image_path)
            lines_original = [item[1] for item in (ocr_result or []) if len(item) > 1 and isinstance(item[1], str)]
            parsed = parse_nutrition_from_ocr_lines(lines_original)
            for column, value in parsed.items():
                df.at[row_index, column] = value
        except Exception as exc:
            LOGGER.warning("OCR fallido en fila %d (%s): %s", row_index + 1, image_path, exc)
        processed += 1
        if processed % LOG_EVERY_PRODUCTS == 0 or row_index + 1 == total_rows:
            LOGGER.info("Avance OCR nutricional: %d/%d", row_index + 1, total_rows)

    df.to_excel(excel_path, index=False)
    LOGGER.info("Columnas nutricionales OCR actualizadas en %s", excel_path)


def download_image_file(url: str, output_path: str, retries: int = 3) -> tuple[str, Optional[str]]:
    """Download one image with retry logic."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return "skipped", None

    last_error: Optional[str] = None
    for attempt in range(retries):
        try:
            with requests.get(
                url,
                timeout=35,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as response:
                response.raise_for_status()
                with open(output_path, "wb") as file_obj:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            file_obj.write(chunk)
            return "ok", None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.4 * (attempt + 1))
    return "error", last_error


def download_images_from_excel(
    excel_path: str,
    images_dir: str,
    max_images_per_row: int = MAX_IMAGES_PER_ROW,
    max_workers: int = MAX_IMAGE_DOWNLOAD_WORKERS,
) -> None:
    """Download up to max_images_per_row images from photo_urls for each Excel row.

    Also updates the Excel with `nutrition_image_file`, pointing to `_img2`
    when that image exists for a row.
    """
    LOGGER.info("Descargando imagenes desde %s", excel_path)
    os.makedirs(images_dir, exist_ok=True)
    df = pd.read_excel(excel_path)
    if "photo_urls" not in df.columns:
        LOGGER.warning("No existe columna 'photo_urls' en el Excel. Se omite la descarga.")
        return
    if "product_id" not in df.columns:
        LOGGER.warning("No existe columna 'product_id' en el Excel. Se omite la descarga.")
        return

    download_tasks: List[tuple[str, str]] = []
    nutrition_targets: Dict[int, str] = {}
    for row_index, row in enumerate(df.itertuples(index=False), start=1):
        product_id = normalize_product_id(getattr(row, "product_id")) or "noid"
        urls = split_photo_urls(getattr(row, "photo_urls"))[:max_images_per_row]
        for img_index, url in enumerate(urls, start=1):
            ext = os.path.splitext(urlparse(url).path)[1].lower() or ".jpg"
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            file_name = f"row{row_index:05d}_pid{product_id}_img{img_index}{ext}"
            output_path = os.path.join(images_dir, file_name)
            download_tasks.append((url, output_path))
            if img_index == 2:
                nutrition_targets[row_index] = output_path

    LOGGER.info("Descargas planificadas: %d", len(download_tasks))
    if download_tasks:
        ok_count = 0
        skipped_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_image_file, url, output_path) for url, output_path in download_tasks]
            total = len(futures)
            for completed, future in enumerate(as_completed(futures), start=1):
                status, _ = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1

                if completed % LOG_EVERY_DOWNLOADS == 0 or completed == total:
                    LOGGER.info(
                        "Avance descargas: %d/%d (ok=%d, omitidas=%d, errores=%d)",
                        completed,
                        total,
                        ok_count,
                        skipped_count,
                        error_count,
                    )

        LOGGER.info(
            "Descarga finalizada en %s (ok=%d, omitidas=%d, errores=%d)",
            images_dir,
            ok_count,
            skipped_count,
            error_count,
        )

    nutrition_column: List[Optional[str]] = []
    for row_index in range(1, len(df) + 1):
        nutrition_path = nutrition_targets.get(row_index)
        if nutrition_path and os.path.exists(nutrition_path) and os.path.getsize(nutrition_path) > 0:
            nutrition_column.append(nutrition_path)
        else:
            nutrition_column.append(None)

    df["nutrition_image_file"] = nutrition_column
    df.to_excel(excel_path, index=False)
    LOGGER.info(
        "Columna nutrition_image_file actualizada (%d/%d con imagen)",
        sum(1 for value in nutrition_column if value),
        len(nutrition_column),
    )


def fetch_json(endpoint: str) -> Dict:
    """Retrieve JSON data from a Mercadona API endpoint with retry logic."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(5):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            if attempt == 4:
                LOGGER.error("Error definitivo al consultar %s: %s", url, exc)
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
            wait_seconds = 2 ** attempt
            LOGGER.warning(
                "Error consultando %s (intento %d/5): %s. Reintentando en %s s",
                url,
                attempt + 1,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue
        return resp.json()
    raise RuntimeError(f"Unreachable code for {url}")


def parse_product(product_id: str, category_path: List[str]) -> ProductInfo:
    """Fetch and parse a single product into our dataclass."""
    data = fetch_json(f"/products/{product_id}/")
    category, subcat, subsub = (category_path + [None, None, None])[:3]
    name = data.get("display_name")
    packaging = data.get("packaging")
    price_info = data.get("price_instructions", {}) or {}
    unit_size = price_info.get("unit_size")
    size_format = price_info.get("size_format")
    price_unit = None
    price_bulk = None
    try:
        price_unit = float(price_info.get("unit_price")) if price_info.get("unit_price") else None
    except (TypeError, ValueError):
        price_unit = None
    try:
        price_bulk = float(price_info.get("bulk_price")) if price_info.get("bulk_price") else None
    except (TypeError, ValueError):
        price_bulk = None
    thumbnail_url, photo_urls = extract_image_urls(data)
    nutrition = data.get("nutrition_information", {}) or {}
    ingredients = nutrition.get("ingredients")
    allergens = nutrition.get("allergens")
    return ProductInfo(
        category=category,
        subcategory=subcat,
        subsubcategory=subsub,
        product_id=product_id,
        product_name=name,
        packaging=packaging,
        unit_size=unit_size,
        size_format=size_format,
        price_unit=price_unit,
        price_bulk=price_bulk,
        thumbnail_url=thumbnail_url,
        photo_urls=photo_urls,
        ingredients=ingredients,
        allergens=allergens,
    )


def scrape_mercadona() -> pd.DataFrame:
    """Scrape all categories, products and nutrition/price info into a DataFrame."""
    categories_data = fetch_json("/categories/")
    categories = categories_data.get("results", [])
    LOGGER.info("Categorias principales encontradas: %d", len(categories))
    items: List[ProductInfo] = []
    processed_products = 0

    for category_index, cat_entry in enumerate(categories, start=1):
        category_name = cat_entry.get("name") or "Sin nombre"
        subcategories = cat_entry.get("categories", []) or []
        LOGGER.info(
            "Categoria %d/%d: %s (%d subcategorias)",
            category_index,
            len(categories),
            category_name,
            len(subcategories),
        )

        for subcat_index, subcat in enumerate(subcategories, start=1):
            subcat_id = subcat.get("id")
            subcat_name = subcat.get("name") or "Sin nombre"
            if not subcat_id:
                LOGGER.warning("Subcategoria sin id en %s: %s", category_name, subcat_name)
                continue
            LOGGER.info(
                "Procesando subcategoria %d/%d: %s",
                subcat_index,
                len(subcategories),
                subcat_name,
            )
            subcat_data = fetch_json(f"/categories/{subcat_id}/")
            # Nested subcategories
            for sub_subcat in subcat_data.get("categories", []) or []:
                sub_subcat_name = sub_subcat.get("name") or "Sin nombre"
                nested_products = sub_subcat.get("products", []) or []
                if nested_products:
                    LOGGER.info("  Sub-subcategoria %s: %d productos", sub_subcat_name, len(nested_products))
                for prod in nested_products:
                    prod_id = prod.get("id")
                    if prod_id:
                        info = parse_product(str(prod_id), [category_name, subcat_name, sub_subcat_name])
                        items.append(info)
                        processed_products += 1
                        if processed_products % LOG_EVERY_PRODUCTS == 0:
                            LOGGER.info("Avance: %d productos procesados", processed_products)
            # Products at the subcategory root
            root_products = subcat_data.get("products", []) or []
            if root_products:
                LOGGER.info("  Productos directos en %s: %d", subcat_name, len(root_products))
            for prod in root_products:
                prod_id = prod.get("id")
                if prod_id:
                    info = parse_product(str(prod_id), [category_name, subcat_name, None])
                    items.append(info)
                    processed_products += 1
                    if processed_products % LOG_EVERY_PRODUCTS == 0:
                        LOGGER.info("Avance: %d productos procesados", processed_products)
    # Convert dataclasses to a DataFrame
    LOGGER.info("Scraping completado: %d productos totales", len(items))
    records = [vars(item) for item in items]
    return pd.DataFrame(records)


def update_excel_images(excel_path: str) -> None:
    """Update image link columns in an existing Excel without re-scraping all data."""
    LOGGER.info("Actualizando solo enlaces de imagen en: %s", excel_path)
    df = pd.read_excel(excel_path)
    if "product_id" not in df.columns:
        raise ValueError("El Excel no contiene la columna obligatoria 'product_id'.")

    if "thumbnail_url" not in df.columns:
        df["thumbnail_url"] = None
    if "photo_urls" not in df.columns:
        df["photo_urls"] = None

    normalized_ids = [normalize_product_id(value) for value in df["product_id"]]
    unique_ids = sorted({product_id for product_id in normalized_ids if product_id})
    LOGGER.info("Productos unicos a consultar: %d", len(unique_ids))

    image_map: Dict[str, tuple[Optional[str], Optional[str]]] = {}
    errors = 0
    for index, product_id in enumerate(unique_ids, start=1):
        try:
            data = fetch_json(f"/products/{product_id}/")
            image_map[product_id] = extract_image_urls(data)
        except Exception as exc:
            errors += 1
            LOGGER.warning("No se pudieron obtener imagenes para %s: %s", product_id, exc)
            image_map[product_id] = (None, None)
        if index % LOG_EVERY_PRODUCTS == 0 or index == len(unique_ids):
            LOGGER.info("Avance imagenes: %d/%d", index, len(unique_ids))

    thumbnails: List[Optional[str]] = []
    photos: List[Optional[str]] = []
    for row_index, product_id in enumerate(normalized_ids):
        existing_thumbnail = df.at[row_index, "thumbnail_url"]
        existing_photo_urls = df.at[row_index, "photo_urls"]
        new_thumbnail, new_photo_urls = image_map.get(product_id, (None, None)) if product_id else (None, None)
        thumbnails.append(new_thumbnail or (None if pd.isna(existing_thumbnail) else existing_thumbnail))
        photos.append(new_photo_urls or (None if pd.isna(existing_photo_urls) else existing_photo_urls))

    df["thumbnail_url"] = thumbnails
    df["photo_urls"] = photos
    df.to_excel(excel_path, index=False)
    LOGGER.info("Excel actualizado: %s (%d filas, %d errores)", excel_path, len(df), errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scraper de productos de Mercadona.")
    parser.add_argument(
        "--update-images-only",
        action="store_true",
        help="Actualiza solo columnas de imagen en un Excel existente, sin re-scrapear categorias.",
    )
    parser.add_argument(
        "--excel-path",
        default="mercadona_data.xlsx",
        help="Ruta del Excel de salida (o entrada/salida en --update-images-only).",
    )
    parser.add_argument(
        "--images-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"),
        help="Carpeta destino para descargar imagenes desde la columna photo_urls.",
    )
    parser.add_argument(
        "--extract-nutrition-ocr",
        action="store_true",
        help="Ejecuta OCR sobre nutrition_image_file y agrega columnas nutricionales al Excel.",
    )
    parser.add_argument(
        "--nutrition-ocr-only",
        action="store_true",
        help="Solo ejecuta OCR nutricional sobre nutrition_image_file, sin actualizar API ni descargar imagenes.",
    )
    parser.add_argument(
        "--ocr-max-rows",
        type=int,
        default=None,
        help="Limita el numero de filas procesadas por OCR nutricional (debug).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.nutrition_ocr_only:
        extract_nutrition_to_excel(args.excel_path, max_rows=args.ocr_max_rows)
        return

    if args.update_images_only:
        update_excel_images(args.excel_path)
        download_images_from_excel(args.excel_path, args.images_dir)
        if args.extract_nutrition_ocr:
            extract_nutrition_to_excel(args.excel_path, max_rows=args.ocr_max_rows)
        return

    LOGGER.info("Iniciando scraper de Mercadona")
    df = scrape_mercadona()
    # Strip HTML tags from ingredient and allergen info
    df["Ingredientes"] = df["ingredients"].fillna("").str.replace(r"<[^>]+>", "", regex=True)
    df["Alérgenos"] = df["allergens"].fillna("").str.replace(r"<[^>]+>", "", regex=True)
    df.drop(columns=["ingredients", "allergens"], inplace=True)
    output_path = args.excel_path
    df.to_excel(output_path, index=False)
    LOGGER.info("Guardado: %s, %d productos extraidos", output_path, len(df))
    download_images_from_excel(output_path, args.images_dir)
    if args.extract_nutrition_ocr:
        extract_nutrition_to_excel(output_path, max_rows=args.ocr_max_rows)


if __name__ == "__main__":
    main()
