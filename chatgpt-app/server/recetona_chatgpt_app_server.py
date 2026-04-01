#!/usr/bin/env python3.12
"""Servidor base de la ChatGPT App de RecetONA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from recetona.mcp_app import create_mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Servidor local de la ChatGPT App de RecetONA para pruebas "
            "con ChatGPT Apps."
        )
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="Transporte MCP (recomendado: streamable-http para ChatGPT).",
    )
    args = parser.parse_args()
    mcp = create_mcp()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
