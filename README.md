# RecetONA

Proyecto para extraer, enriquecer y consultar datos de productos de Mercadona usando:
- `mercadona_scraper_script.py` para scraping y actualización del dataset.
- `local_rag_server.py` para exponer el motor RAG local por HTTP.
- `recetona_mcp_server.py` para exponerlo como servidor MCP.

## Requisitos

- Python 3.12
- Un entorno virtual local `.venv`
- `OPENAI_API_KEY` en `.env` para las consultas RAG

## Configuración

1. Crea el entorno virtual:

```bash
python3.12 -m venv .venv
```

2. Actívalo:

```bash
source .venv/bin/activate
```

3. Instala dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Crea tu archivo de entorno:

```bash
cp .env.example .env
```

5. Añade tu clave:

```env
OPENAI_API_KEY=tu_clave_aqui
```

## Ejecución desde terminal

### Scraper

```bash
source .venv/bin/activate
python mercadona_scraper_script.py
```

Ver opciones:

```bash
python mercadona_scraper_script.py -h
```

### API local HTTP

```bash
source .venv/bin/activate
python local_rag_server.py
```

Endpoints:
- `GET /health`
- `POST /chat`

### Frontend demo

Con la API local arrancada en otra terminal:

```bash
cd frontend
python -m http.server 8080
```

Luego abre `http://127.0.0.1:8080/main.html`.

### MCP

```bash
source .venv/bin/activate
python recetona_mcp_server.py --transport stdio
```

Registro en Codex:

```toml
[mcp_servers.RecetONA]
command = "/home/wencm/RecetONA/.venv/bin/python"
args = ["/home/wencm/RecetONA/recetona_mcp_server.py", "--transport", "stdio"]
cwd = "/home/wencm/RecetONA"
startup_timeout_sec = 45
tool_timeout_sec = 240
enabled = true
```

## Notebook

El notebook sigue existiendo para exploración, pero la recomendación operativa es ejecutar el proyecto desde `.venv`. Si lo usas, ábrelo con el kernel del `.venv`.

## Notas

- `images/`, `rag_cache/`, `.venv/`, `.env` y `AGENTS.md` están excluidos por `.gitignore`.
- Ya no hace falta instalar dependencias en `lib/`.
