from __future__ import annotations

import math
import os
import re
import unicodedata
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    return variants


def tokenize(text: Any) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", normalize_text(text))
    output: list[str] = []
    for token in raw:
        output.extend(sorted(token_variants(token)))
    return output


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except Exception:
        return None


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def numeric_text(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return clean_text(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def normalize_product_id(raw_value: Any) -> str | None:
    if pd.isna(raw_value):
        return None
    if isinstance(raw_value, float) and raw_value.is_integer():
        return str(int(raw_value))
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return None
    try:
        as_float = float(value)
    except Exception:
        return value
    return str(int(as_float)) if as_float.is_integer() else f"{as_float:g}"


def split_photo_urls(raw_value: Any) -> list[str]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [url.strip() for url in re.split(r"\s*\|\s*|\s*\n+\s*", text) if url.strip()]


def image_size_score(url: str | None) -> int:
    if not isinstance(url, str) or not url:
        return -1
    try:
        query = parse_qs(urlparse(url).query)
        height = int(query.get("h", ["0"])[0])
        width = int(query.get("w", ["0"])[0])
        if height > 0 and width > 0:
            return height * width
    except Exception:
        return 0
    return 0


def extract_image_urls(product_data: dict[str, Any]) -> tuple[str | None, str | None]:
    original_thumbnail_url = product_data.get("thumbnail")
    photos = product_data.get("photos", []) or []
    photo_urls_list: list[str] = []
    seen: set[str] = set()

    for photo in photos:
        if isinstance(photo, dict):
            candidates = [photo.get(key) for key in ("thumbnail", "regular", "zoom", "url")]
            best_url = None
            best_score = -1
            for candidate in candidates:
                score = image_size_score(candidate)
                if score > best_score:
                    best_url = candidate
                    best_score = score
            if isinstance(best_url, str) and best_url and best_url not in seen:
                seen.add(best_url)
                photo_urls_list.append(best_url)
        elif isinstance(photo, str) and photo and photo not in seen:
            seen.add(photo)
            photo_urls_list.append(photo)

    photo_urls = " | ".join(photo_urls_list) if photo_urls_list else None
    thumbnail_url = photo_urls_list[0] if photo_urls_list else original_thumbnail_url
    return thumbnail_url, photo_urls


def format_money(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.2f}"


def format_number(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def path_exists_and_nonempty(path: str | os.PathLike[str]) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0
