.DEFAULT_GOAL := help
PYTHON ?= python
COMPLIANCE ?= scripts/check_compliance.py

.PHONY: help install format lint types test check clean

help: ## Afficher les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Installer le projet et les dépendances de développement
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

format: ## Formater le code
	ruff format src tests scripts
	ruff check --fix src tests scripts

lint: ## Vérifier le style sans modifier les fichiers
	ruff check src tests scripts
	ruff format --check src tests scripts

types: ## Vérifier les types
	mypy src

test: ## Lancer la suite de tests
	pytest

check: ## Contrôle de conformité complet (structure, code, artefacts)
	$(PYTHON) $(COMPLIANCE) --repo . --report

clean: ## Supprimer les caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
