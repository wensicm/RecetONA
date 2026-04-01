#!/usr/bin/env python3.12
from __future__ import annotations

import sys
from pathlib import Path

from mangum import Mangum

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

for candidate_path in (BASE_DIR, SRC_DIR):
    resolved_path = str(candidate_path)
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

from recetona.mcp_app import build_streamable_http_app


def _build_handler() -> Mangum:
    application = build_streamable_http_app(
        json_response=True,
        stateless_http=True,
    )
    return Mangum(application, lifespan="auto")


def lambda_handler(event: dict, context: object) -> dict:
    handler = _build_handler()
    return handler(event, context)
