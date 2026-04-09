# Contribuir a RecetONA

Gracias por tomarte el tiempo de revisar o mejorar el proyecto.

## Principios

- Mantén los cambios acotados a una parte concreta del repo.
- No mezcles refactors estructurales con cambios de producto.
- No subas secretos, claves ni caches locales.
- Si tocas código Python, mantén compatibilidad con Python `3.12`.

## Mapa rápido

- `mcp/`: backend operativo principal
- `chatgpt-app/`: app de ChatGPT y widget
- `docs/`: documentación de arquitectura y navegación

## Configuración local

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
  -r mcp/requirements.txt \
  -r chatgpt-app/server/requirements.txt
```

Variables de entorno para desarrollo:

```bash
export OPENAI_API_KEY="tu_clave"
```

Si quieres usar `.env` locales, créalos manualmente en el subárbol que estés
tocando. El repo no incluye plantillas `.env.example`.

## Estilo

- Formato Python: `black -l 79`
- Documentación y comentarios técnicos: en español
- Nombres de variables: sin abreviaciones innecesarias

## Validación mínima antes de abrir cambios

```bash
.venv/bin/black -l 79 --check \
  mcp/src mcp/*.py mcp/tests \
  chatgpt-app/server/src chatgpt-app/server/*.py chatgpt-app/server/tests

.venv/bin/pytest mcp/tests chatgpt-app/server/tests
```

Nota: los tests de `mcp/` y `chatgpt-app/` deben ejecutarse en dos invocaciones
separadas porque comparten nombres de módulos de test y un `pytest` único puede
provocar `import file mismatch`.

Si has tocado despliegues Lambda, añade también:

```bash
sam validate --template-file mcp/lambda/recetona_mcp_api/template.yaml
sam validate --template-file chatgpt-app/lambda/recetona_chatgpt_app_api/template.yaml
```

## Pull requests

Un PR bueno para este repo debería:

- explicar qué superficie toca (`mcp` o `chatgpt-app`)
- justificar cualquier duplicación en `lambda/`
- incluir validación ejecutada
- mantener README y docs al día si cambia la arquitectura visible

## Alcance

Si quieres proponer una reestructuración grande del repo, abre el cambio en
dos pasos:

1. mejora documental y de navegación
2. cambio físico de carpetas o fuente de verdad

Así se reduce el riesgo de romper despliegues o rutas locales.
