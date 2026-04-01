# RecetONA

Proyecto para extraer, enriquecer y consultar datos de productos de Mercadona usando:
- `mercadona_scraper_script.py` para scraping y actualización del dataset.
- `local_rag_server.py` para exponer el motor RAG local por HTTP.
- `recetona_mcp_server.py` para exponerlo como servidor MCP.

Esta carpeta `mcp/` es la implementación operativa actual del proyecto.

## Requisitos

- Python 3.12
- Un entorno virtual local `.venv`
- `OPENAI_API_KEY` en `.env` para las consultas RAG

## Configuración

1. Crea el entorno virtual:

```bash
cd /Users/wensicm/Repositorios/RecetONA
python3.12 -m venv .venv
```

2. Actívalo:

```bash
cd /Users/wensicm/Repositorios/RecetONA
source .venv/bin/activate
```

3. Instala dependencias:

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Crea tu archivo de entorno:

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
cp .env.example .env
```

5. Añade tu clave:

```env
OPENAI_API_KEY=tu_clave_aqui
```

## Ejecución desde terminal

### Scraper

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
source ../.venv/bin/activate
python mercadona_scraper_script.py
```

Ver opciones:

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
python mercadona_scraper_script.py -h
```

### API local HTTP

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
source ../.venv/bin/activate
python local_rag_server.py
```

Endpoints:
- `POST /chat`

### Regenerar `rag_cache`

Cada fila de `mercadona_data.xlsx` se indexa como un chunk independiente.
Para regenerar `rag_cache/chunks.csv` y `rag_cache/embeddings.npy` con
`text-embedding-3-large` y 18 workers:

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
source ../.venv/bin/activate
caffeinate -dimsu python build_rag_cache.py --model text-embedding-3-large --workers 18
```

El builder mantiene checkpoints en `rag_cache/embeddings.partial/` y puede
reanudar automáticamente si se interrumpe.

### Frontend demo

Con la API local arrancada en otra terminal:

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp/frontend
python3 -m http.server 8080
```

Luego abre `http://127.0.0.1:8080/main.html`.

### MCP

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
source ../.venv/bin/activate
python recetona_mcp_server.py --transport stdio
```

Registro en Codex:

```toml
[mcp_servers.RecetONA]
command = "/home/wencm/RecetONA/.venv/bin/python"
args = ["/home/wencm/RecetONA/mcp/recetona_mcp_server.py", "--transport", "stdio"]
cwd = "/home/wencm/RecetONA/mcp"
startup_timeout_sec = 45
tool_timeout_sec = 240
enabled = true
```

## Notebook

El notebook sigue existiendo para exploración, pero la recomendación operativa es ejecutar el proyecto desde `.venv`. Si lo usas, ábrelo con el kernel del `.venv`.

## Notas

- `images/`, `rag_cache/`, `.venv/`, `.env` y `AGENTS.md` están excluidos por `.gitignore`.
- Ya no hace falta instalar dependencias en `lib/`.
