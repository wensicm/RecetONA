# RecetONA

Proyecto para extraer, enriquecer y consultar datos de productos de Mercadona usando:
- `mercadona_scraper_script.py` para scraping/actualización de datos.
- `mercadona_rag_notebook.ipynb` para consultas tipo RAG con OpenAI sobre `mercadona_data.xlsx`.

## Estructura

- `mercadona_scraper_script.py`: script principal de scraping y procesamiento.
- `mercadona_data.xlsx`: dataset de productos.
- `mercadona_rag_notebook.ipynb`: notebook para consultas y generación de recetas/coste.
- `requirements.txt`: dependencias Python.
- `lib/`: instalación local de dependencias (target de `pip`).
- `images/`: imágenes descargadas de productos.
- `rag_cache/`: caché de embeddings/chunks para RAG.

## Requisitos

- Python 3.12
- API key de OpenAI

## Configuración

1. Crear archivo de entorno:

```bash
cp .env.example .env
```

2. Añadir tu clave:

```env
OPENAI_API_KEY=tu_clave_aqui
```

3. Instalar dependencias en `lib`:

```bash
/usr/bin/python3 -m pip install --upgrade --target ./lib -r requirements.txt
```

## Uso del scraper

Ejecuta el script principal:

```bash
/usr/bin/python3 mercadona_scraper_script.py
```

Para ver opciones disponibles:

```bash
/usr/bin/python3 mercadona_scraper_script.py -h
```

## Uso del notebook RAG

1. Abrir `mercadona_rag_notebook.ipynb`.
2. Ejecutar las celdas en orden.
3. La primera celda instala automáticamente `requirements.txt` en `lib`.
4. Lanza tus preguntas sobre productos/recetas y costes.

## Empaquetado como MCP (`@RecetONA`)

Este repo incluye `recetona_mcp_server.py`, un servidor MCP local (Python 3.12) con tools:

- `query_recipe(pregunta)`: usa el motor RAG del notebook y devuelve el texto final listo para el usuario.
- `search(query, limit)`: busca productos en el catálogo.
- `fetch(id)`: recupera detalle completo de un producto.

### 1) Instalar dependencias en `lib`

```bash
/usr/bin/python3.12 -m pip install --upgrade --target ./lib -r requirements.txt
```

### 2) Registrar MCP en Codex (`config.toml`)

En `~/.codex/config.toml` (o `.codex/config.toml` del proyecto):

```toml
[mcp_servers.RecetONA]
command = "/usr/bin/python3.12"
args = ["/home/wencm/RecetONA/recetona_mcp_server.py", "--transport", "stdio"]
cwd = "/home/wencm/RecetONA"
startup_timeout_sec = 45
tool_timeout_sec = 240
enabled = true
```

### 3) Usarlo en chat

- En clientes que soportan menciones, invócalo como `@RecetONA`.
- Si el cliente no soporta menciones, pide explícitamente usar la tool `query_recipe`.

Ejemplo:

```text
@RecetONA dame una receta de lentejas para 4 personas y el precio de compra
```

## Notas

- Este repositorio usa rutas locales y caché (`rag_cache/`) para acelerar consultas.
- `lib/`, `images/`, `rag_cache/`, `.env` y `AGENTS.md` están excluidos por `.gitignore`.

## Propiedad intelectual

La propiedad intelectual de este repositorio (código, notebooks, documentación y estructura del proyecto) pertenece al creador del repositorio.  
Todos los derechos reservados.
