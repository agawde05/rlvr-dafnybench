# ============================================================================
# RLVR-DafnyBench Makefile
# ----------------------------------------------------------------------------
# Unified developer interface for Dockerized + uv-managed environment
# ============================================================================
# Usage examples:
#   make build          # Build Docker image
#   make shell          # Enter container shell
#   make train          # Run training script
#   make eval           # Run evaluation script
#   make format         # Autoformat with ruff
#   make lint           # Lint with ruff
#   make test           # Run pytest suite
#   make rebuild        # Clean volumes and rebuild from scratch
#   make run            # Full mock pipeline (data + train + eval)
#   make help           # Show all available targets
# ============================================================================

# Compose file
COMPOSE_FILE := docker-compose.dev.yml
SERVICE := rlvr
CONTAINER := rlvr-dev

# -------- Core Docker Targets --------

## Build the dev container image
build:
	docker compose -f $(COMPOSE_FILE) build

## Start container and attach shell
shell:
	docker compose -f $(COMPOSE_FILE) up -d $(SERVICE)
	docker exec -it $(CONTAINER) bash

## Bring up container in detached mode
up:
	docker compose -f $(COMPOSE_FILE) up -d

## Stop and remove containers (keep volumes)
down:
	docker compose -f $(COMPOSE_FILE) down

## Full clean rebuild (remove volumes + rebuild image)
rebuild:
	docker compose -f $(COMPOSE_FILE) down -v
	docker compose -f $(COMPOSE_FILE) build --no-cache
	make shell

# -------- Development Commands --------

## Run training (mock or real, depending on config)
train:
	uv run scripts/train_model.py --config config/train_config.yaml

## Run evaluation
eval:
	uv run scripts/evaluate_model.py

## Fetch mock dataset
data:
	uv run scripts/get_data.py

## Run data → train → eval sequentially
run: data train eval

# -------- Quality & Testing --------

## Format code automatically with Ruff
format:
	ruff check --fix .

## Lint code (no fixes)
lint:
	ruff check .

## Run unit tests
test:
	uv run pytest -q

# -------- Utility --------

## Display help
help:
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
