.PHONY: help setup docker-up docker-down docker-logs docker-restart ingest serve ui query benchmark test clean

# Default Target
.DEFAULT_GOAL := help

# Detect Docker Compose Command (supports modern 'docker compose' and legacy 'docker-compose')
DOCKER_COMPOSE := $(shell command -v docker-compose 2>/dev/null || echo "docker compose")

help:
	@echo "========================================================================="
	@echo "Knowledge Graph RAG for Enterprise Data - Makefile Commands"
	@echo "========================================================================="
	@echo "Environment & Docker Commands:"
	@echo "  make setup          Install Python dependencies and frontend packages"
	@echo "  make docker-up      Launch Neo4j (5.x) & pgvector (PostgreSQL 16) containers"
	@echo "  make docker-down    Stop and remove Docker containers"
	@echo "  make docker-logs    Tail logs for Neo4j and pgvector containers"
	@echo "  make docker-restart Restart Docker containers"
	@echo ""
	@echo "Application & Execution Commands:"
	@echo "  make ingest         Run budget-constrained corpus ingestion into Neo4j & pgvector"
	@echo "  make serve          Start FastAPI REST API server on http://localhost:8000"
	@echo "  make ui             Launch React Vite Frontend on http://localhost:5173"
	@echo "  make query Q='...'  Execute a natural language query via CLI"
	@echo "                      Example: make query Q='Which packages affect EU CRA compliance?'"
	@echo "  make benchmark      Run stratified 50-item comparative benchmark"
	@echo ""
	@echo "Testing & Utility Commands:"
	@echo "  make test           Run unit test suite"
	@echo "  make clean          Clean local Python bytecode and temporary caches"
	@echo "========================================================================="

setup:
	pip install -r requirements.txt
	cd frontend && npm install

docker-up:
	$(DOCKER_COMPOSE) up -d
	@echo "Waiting for Neo4j and pgvector services to initialize..."

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-restart:
	$(DOCKER_COMPOSE) restart

ingest:
	PYTHONPATH=. python3 scripts/ingest_corpus.py

serve:
	PYTHONPATH=. python3 -m src.presentation.cli.main serve

ui:
	cd frontend && npm run dev

query:
	@if [ -z "$(Q)" ]; then \
		echo "Error: Please specify query string using Q='your question'"; \
		echo "Example: make query Q='Which packages affect EU CRA compliance for Acme EU GmbH?'"; \
		exit 1; \
	fi
	PYTHONPATH=. python3 -m src.presentation.cli.main query "$(Q)"

benchmark:
	PYTHONPATH=. python3 -m src.presentation.cli.main benchmark data/benchmark_questions.json

test:
	PYTHONPATH=. python3 -m unittest discover -s tests/unit

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf frontend/dist frontend/node_modules/.vite
