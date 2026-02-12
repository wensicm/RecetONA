#!/usr/bin/env python3.12
import argparse
import json
import logging
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR / "lib"
NOTEBOOK_PATH = BASE_DIR / "mercadona_rag_notebook.ipynb"
ENV_PATH = BASE_DIR / ".env"

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _extract_code_cells(notebook_path: Path) -> list[str]:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            code_cells.append(source)
    return code_cells


def _select_runtime_cells(code_cells: list[str]) -> list[str]:
    selected = []
    markers = [
        "EMBED_MODEL =",
        "def _clean(v):",
        "client = OpenAI",
        "embeddings = ensure_embeddings(chunks)",
        "def ask_agent(",
    ]
    for marker in markers:
        match = next((c for c in code_cells if marker in c), None)
        if not match:
            raise RuntimeError(f"No se encontro la celda requerida con marcador: {marker}")
        selected.append(match)
    return selected


def build_notebook_runtime(notebook_path: Path) -> dict:
    cells = _extract_code_cells(notebook_path)
    runtime_cells = _select_runtime_cells(cells)

    code = "\n\n".join(runtime_cells)
    code = code.replace("/home/wencm/Alimentación", str(BASE_DIR))

    namespace: dict = {"__name__": "__recetona_notebook_runtime__"}
    exec(compile(code, str(notebook_path), "exec"), namespace, namespace)
    if "ask_agent" not in namespace:
        raise RuntimeError("No se pudo cargar ask_agent desde el notebook.")
    return namespace


class NotebookRagService:
    def __init__(self, notebook_path: Path):
        self._lock = threading.Lock()
        self._ns = build_notebook_runtime(notebook_path)
        self._ask_agent = self._ns["ask_agent"]

    def ask(self, message: str) -> dict:
        with self._lock:
            result = self._ask_agent(
                message,
                top_k=35,
                retrieval_mode="hybrid",
                alpha=0.65,
                recipe_mode="auto",
                use_ingredient_tool=True,
                candidates_per_ingredient=12,
            )

        return {
            "answer": str(result.get("answer", "")).strip(),
            "block_1": str(result.get("block_1", "")).strip(),
            "block_2": str(result.get("block_2", "")).strip(),
            "block_3": str(result.get("block_3", "")).strip(),
        }


class LocalApiHandler(BaseHTTPRequestHandler):
    service: NotebookRagService | None = None

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True, "service": "recetona-local-rag"})
            return
        self._send_json(404, {"error": "Ruta no encontrada"})

    def do_POST(self):
        if self.path != "/chat":
            self._send_json(404, {"error": "Ruta no encontrada"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "JSON invalido"})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json(400, {"error": "Falta el campo 'message'"})
            return

        if self.service is None:
            self._send_json(503, {"error": "Servicio no inicializado"})
            return

        try:
            response = self.service.ask(message)
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(
                500,
                {
                    "error": f"Fallo ejecutando ask_agent: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor local para conectar frontend con ask_agent del notebook.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    load_env_file(ENV_PATH)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Falta OPENAI_API_KEY. Define la clave en {ENV_PATH} o en variables de entorno.")

    logging.info("Inicializando runtime desde notebook: %s", NOTEBOOK_PATH)
    service = NotebookRagService(NOTEBOOK_PATH)
    LocalApiHandler.service = service

    server = ThreadingHTTPServer((args.host, args.port), LocalApiHandler)
    logging.info("API local escuchando en http://%s:%d", args.host, args.port)
    logging.info("Endpoints: GET /health, POST /chat")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Cerrando servidor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
