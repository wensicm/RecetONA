# RecetONA MCP

`mcp/` es la implementación operativa actual del backend de RecetONA.

Aquí viven:

- el scraper del catálogo
- la generación del índice RAG
- el servidor HTTP local
- el servidor MCP
- los tests del backend
- el despliegue Lambda del MCP base

## Requisitos

- Python `3.12`
- entorno virtual local en `../.venv`
- `OPENAI_API_KEY` para consultas RAG

## Configuración

Desde la raíz del repo:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r mcp/requirements.txt
cp mcp/.env.example mcp/.env
```

## Flujos habituales

### Scraper

```bash
cd mcp
../.venv/bin/python mercadona_scraper_script.py
```

### API local HTTP

```bash
cd mcp
../.venv/bin/python local_rag_server.py
```

Endpoint principal:

- `POST /chat`

### Regenerar `rag_cache`

Cada fila de `mercadona_data.xlsx` se indexa como un chunk independiente.

```bash
cd mcp
caffeinate -dimsu ../.venv/bin/python build_rag_cache.py \
  --model text-embedding-3-large \
  --workers 18
```

El builder mantiene checkpoints en `rag_cache/embeddings.partial/` y puede
reanudar automáticamente.

### Frontend demo

Con la API local arrancada en otra terminal:

```bash
cd mcp/frontend
python3 -m http.server 8080
```

Luego abre `http://127.0.0.1:8080/main.html`.

### Servidor MCP

```bash
cd mcp
../.venv/bin/python recetona_mcp_server.py --transport stdio
```

## Estructura útil

- `src/recetona/`: fuente de verdad del backend MCP
- `tests/`: cobertura de lógica principal
- `lambda/recetona_mcp_api/`: snapshot autocontenido para despliegue SAM

## Nota sobre despliegue

La carpeta `lambda/recetona_mcp_api/` no sustituye a `mcp/` como fuente de
verdad. Es un snapshot de despliegue para `sam build --use-container`.
