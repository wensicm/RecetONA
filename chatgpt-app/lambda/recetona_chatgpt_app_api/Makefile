PYTHON_BIN ?= python3.12
BUILD_VENV_DIR := $(ARTIFACTS_DIR)/.build-venv

build-RecetonaChatgptAppFunction:
	"$(PYTHON_BIN)" -m venv "$(BUILD_VENV_DIR)"
	"$(BUILD_VENV_DIR)/bin/python" -m pip install --upgrade pip
	"$(BUILD_VENV_DIR)/bin/python" -m pip install \
		-r requirements-lambda.txt \
		-t "$(ARTIFACTS_DIR)"
	cp app.py "$(ARTIFACTS_DIR)/app.py"
	mkdir -p "$(ARTIFACTS_DIR)/src"
	cp -R src/recetona "$(ARTIFACTS_DIR)/src/"
	mkdir -p "$(ARTIFACTS_DIR)/web"
	cp -R web/public "$(ARTIFACTS_DIR)/web/"
	cp local_rag_server.py "$(ARTIFACTS_DIR)/local_rag_server.py"
	cp mercadona_data.xlsx \
		"$(ARTIFACTS_DIR)/mercadona_data.xlsx"
	cp mercadona_rag_notebook.ipynb \
		"$(ARTIFACTS_DIR)/mercadona_rag_notebook.ipynb"
	if [ -d rag_cache ]; then \
		cp -R rag_cache "$(ARTIFACTS_DIR)/rag_cache"; \
	fi
	rm -rf "$(BUILD_VENV_DIR)"
