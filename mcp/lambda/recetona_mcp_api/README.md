# RecetONA MCP en AWS Lambda

Este despliegue expone el MCP de `RecetONA` como endpoint remoto
`streamable-http` sobre API Gateway HTTP API, con custom domain.

## Archetype

`tool-only`

No se añade widget. Para este caso ChatGPT Apps puede conectarse
directamente al servidor MCP remoto y usar `search`, `fetch` y
`query_recipe`.

## Por qué esta forma

- La guía de Quickstart del Apps SDK indica que el UI es opcional cuando
  solo necesitas tools.
- La guía de `Build your MCP server` recomienda exponer el servidor por
  HTTPS y mantener `structuredContent` y tools bien descritas.
- `RecetONA` ya tiene `search` y `fetch`, que encajan con el patrón de
  app de datos.

## Archivos

- `template.yaml`: stack SAM con Lambda + HTTP API.
- `app.py`: entrypoint Lambda con `Mangum`.
- `requirements-lambda.txt`: dependencias mínimas para el runtime MCP.
- `Makefile`: build SAM de la carpeta autocontenida de la Lambda.
- `src/`, `local_rag_server.py`, `mercadona_data.xlsx` y
  `mercadona_rag_notebook.ipynb`: snapshot local del runtime necesario para
  que `sam build --use-container` funcione sin montar la raíz completa del
  repo dentro del contenedor.

## URL propuesta

`https://api.wensicm.com/recetona/mcp`

No uso `https://wensicm.com/recetona/mcp` porque `wensicm.com` ya sirve la
web principal por CloudFront. Reutilizar el dominio raiz para API Gateway
haria mas fragil el routing de toda la web.

## Build y deploy

```bash
cd /Users/wensicm/Repositorios/RecetONA/lambda/recetona_mcp_api
sam build --use-container
sam deploy --stack-name recetona-mcp --resolve-s3 --capabilities CAPABILITY_IAM
```

## Variables y caches

`query_recipe` usa el runtime RAG del notebook. En Lambda no conviene
generar embeddings en caliente, así que `template.yaml` fija
`RECETONA_REQUIRE_PREBUILT_RAG_CACHE=true`.

Antes de desplegar en serio, deja preparados estos artefactos en la raíz
del repo para que el build los copie si existen:

- `rag_cache/chunks.csv`
- `rag_cache/embeddings.npy`

Si no están, `search` y `fetch` seguirán funcionando desde
`mercadona_data.xlsx`, pero `query_recipe` fallará de forma explícita en
vez de intentar recomputar embeddings dentro de la Lambda.

La plantilla lee `OPENAI_API_KEY` desde SSM Parameter Store usando el
SecureString `OPENAI_API_KEY` por defecto. Si quieres otro nombre, cambia
el parámetro `OpenAiApiKeyParameterName` al desplegar.

## Conexión desde ChatGPT Apps

1. Despliega el stack y copia el output `CustomDomainMcpUrl`.
2. En ChatGPT, activa Developer Mode.
3. Crea una app nueva y pega la URL HTTPS terminada en `/mcp`.
4. Refresca la app después de cambiar tools o metadata.

## Documentación usada

- https://developers.openai.com/apps-sdk/quickstart/
- https://developers.openai.com/apps-sdk/build/mcp-server/
- https://developers.openai.com/apps-sdk/build/chatgpt-ui/
