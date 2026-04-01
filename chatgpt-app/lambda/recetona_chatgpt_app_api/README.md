# RecetONA ChatGPT App en AWS Lambda

Este despliegue expone la `chatgpt-app` de `RecetONA` como endpoint MCP
remoto `streamable-http` sobre API Gateway HTTP API, reutilizando el custom
domain ya existente de `api.wensicm.com`.

## Archetype

`interactive-decoupled`

La app separa:

- `query_recipe_data`: tool de datos
- `render_recipe_widget`: tool de render con widget

`search` y `fetch` se mantienen como tools de catalogo.

## Por que esta forma

- La guia `Build your ChatGPT UI` recomienda el patron desacoplado para que
  ChatGPT no remonte el iframe en cada tool call.
- La guia `Build your MCP server` recomienda versionar la URI del widget,
  devolver `structuredContent` conciso y reservar el template para el
  render tool.
- La guia `Deploy your app` pide un endpoint HTTPS estable y con soporte de
  streaming en el endpoint MCP publico.

## Archivos

- `template.yaml`: stack SAM con Lambda + HTTP API.
- `app.py`: entrypoint Lambda con `Mangum`.
- `requirements-lambda.txt`: dependencias minimas del runtime.
- `Makefile`: build SAM autocontenido.
- `src/`, `web/public/`, `local_rag_server.py`,
  `mercadona_data.xlsx` y `mercadona_rag_notebook.ipynb`: snapshot del
  runtime necesario para que `sam build --use-container` funcione sin montar
  toda la raiz del repo.

## URL propuesta

`https://api.wensicm.com/recetona/app`

Se reutiliza el custom domain ya creado para el MCP base y el API mapping se
crea despues del deploy con AWS CLI para evitar el problema de orden interno
de recursos que introduce SAM con el stage por defecto. El mapping publico
actual cuelga de `recetona/app` y el MCP vive en la raiz de ese base path.

## Build y deploy

```bash
cd /Users/wensicm/Repositorios/RecetONA/chatgpt-app/lambda/recetona_chatgpt_app_api
sam build --use-container
sam deploy --stack-name recetona-chatgpt-app-v3 --resolve-s3 --capabilities CAPABILITY_IAM
API_ID="$(aws cloudformation describe-stacks \
  --stack-name recetona-chatgpt-app-v3 \
  --region eu-west-1 \
  --query 'Stacks[0].Outputs[?OutputKey==`HttpApiBaseUrl`].OutputValue' \
  --output text | sed -E 's#https://([^.]*)\\.execute-api\\..*#\\1#')"
aws apigatewayv2 create-api-mapping \
  --region eu-west-1 \
  --domain-name api.wensicm.com \
  --api-id "$API_ID" \
  --stage '$default' \
  --api-mapping-key recetona/app
```

## Variables y caches

La app reusa la cache RAG ya precalculada del stack MCP principal, leyendo
desde el bucket S3 existente:

- bucket: `recetona-mcp-recetonaragcachebucket-bzqmbddjds7z`
- prefijo: `recetona/rag_cache`

La plantilla lee `OPENAI_API_KEY` desde SSM Parameter Store usando el
SecureString `OPENAI_API_KEY` por defecto.

## Conexion desde ChatGPT Apps

1. Despliega el stack y copia el output `CustomDomainMcpUrl`.
2. En ChatGPT, crea una app nueva y pega la URL HTTPS exacta de la app.
3. Si cambias tools, widget o metadata, refresca la app para que ChatGPT
   recargue el descriptor.

## Documentacion usada

- https://developers.openai.com/apps-sdk/build/mcp-server/
- https://developers.openai.com/apps-sdk/build/chatgpt-ui/
- https://developers.openai.com/apps-sdk/deploy/
