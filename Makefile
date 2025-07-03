VENV_DIR := .venv

help:
	@echo "Available targets:"
	@echo "  install     - Install all dependencies from requirements.txt into existing venv"
	@echo "  reinstall   - Recreate venv and reinstall all dependencies from requirements.txt"

create-env:
	python3 -m venv $(VENV_DIR)

delete-env:
	rm -rf $(VENV_DIR)

install:
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip
	. $(VENV_DIR)/bin/activate && pip install -r requirements.txt

reinstall: delete-env create-env install
