# RecetONA

El repo queda dividido en dos raíces:

- `mcp/`: implementación operativa actual. Aquí está todo lo que ya funciona
  como scraper, RAG local, servidor MCP, frontend demo, tests y Lambda.
- `chatgpt-app/`: scaffold separado para evolucionar RecetONA hacia una
  ChatGPT App con `server/` y `web/`.

## Uso rápido

### Backend MCP actual

```bash
cd /Users/wensicm/Repositorios/RecetONA/mcp
source ../.venv/bin/activate
python recetona_mcp_server.py --transport stdio
```

Más detalle en [mcp/README.md](/Users/wensicm/Repositorios/RecetONA/mcp/README.md).

### Scaffold de ChatGPT App

```bash
cd /Users/wensicm/Repositorios/RecetONA/chatgpt-app/server
source ../../.venv/bin/activate
python recetona_chatgpt_app_server.py --transport streamable-http
```

Más detalle en
[chatgpt-app/README.md](/Users/wensicm/Repositorios/RecetONA/chatgpt-app/README.md).
