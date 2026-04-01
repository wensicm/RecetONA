# RecetONA ChatGPT App

Esta carpeta es el scaffold separado para evolucionar RecetONA hacia una
ChatGPT App con estructura tipo Apps SDK:

- `server/`: servidor MCP reutilizando una copia del runtime RAG y del paquete
  `recetona` que hoy vive en `mcp/`.
- `web/public/`: widget HTML desacoplado del servidor, pensado para renderizar
  dentro de ChatGPT.

## Qué se ha reutilizado desde `mcp/`

Se han copiado estos elementos para aislar el trabajo de la app respecto al
backend MCP ya operativo:

- `server/src/recetona/`
- `server/local_rag_server.py`
- `server/mercadona_data.xlsx`
- `server/mercadona_rag_notebook.ipynb`
- `server/requirements.txt`

## Arquitectura

El server de esta carpeta ya sigue el patron recomendado de Apps SDK para
apps con widget:

- `query_recipe_data`: tool de datos
- `render_recipe_widget`: tool de render
- `search` y `fetch`: tools de catalogo

El widget desacoplado se sirve desde `web/public/recetona-widget.html` y la
Lambda autocontenida vive en:

- `lambda/recetona_chatgpt_app_api/`

## Arranque local

```bash
cd /Users/wensicm/Repositorios/RecetONA/chatgpt-app/server
source ../../.venv/bin/activate
python -m pip install -r requirements.txt
python recetona_chatgpt_app_server.py --transport streamable-http
```

El widget externo que usa esta copia está en:

- [recetona-widget.html](/Users/wensicm/Repositorios/RecetONA/chatgpt-app/web/public/recetona-widget.html)

## Estado

Esto no sustituye al backend de `mcp/`, que sigue vivo en su endpoint propio.
La `chatgpt-app` queda desplegada aparte en:

- [https://api.wensicm.com/recetona/app](https://api.wensicm.com/recetona/app)

Se mantiene la separación para que el MCP base y la app de ChatGPT puedan
evolucionar por caminos distintos sin romperse entre si.
