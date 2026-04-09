# RecetONA ChatGPT App

`chatgpt-app/` contiene la versión de RecetONA pensada para integrarse con
ChatGPT Apps.

## Qué hay aquí

- `server/`: runtime MCP y tools de la app
- `web/public/`: widget HTML desacoplado
- `lambda/recetona_chatgpt_app_api/`: snapshot autocontenido para despliegue
  SAM

## Arquitectura

La app sigue un patrón desacoplado:

- `query_recipe_data`: tool que resuelve datos de receta
- widget HTML: render del resultado estructurado
- `search` y `fetch`: tools de catálogo

La lógica principal vive en:

- `server/src/recetona/`
- `server/local_rag_server.py`
- `web/public/recetona-widget.html`

## Arranque local

Desde la raíz del repo:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r chatgpt-app/server/requirements.txt
cp chatgpt-app/server/.env.example chatgpt-app/server/.env
```

Luego:

```bash
cd chatgpt-app/server
../../.venv/bin/python recetona_chatgpt_app_server.py --transport streamable-http
```

## Endpoint desplegado

La app está desplegada en:

- [https://api.wensicm.com/recetona/app](https://api.wensicm.com/recetona/app)

## Nota sobre `lambda/`

La carpeta `lambda/recetona_chatgpt_app_api/` es un snapshot autocontenido
para `sam build --use-container`. La fuente de verdad para desarrollo sigue
estando en `server/` y `web/public/`.
