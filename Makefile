# -------- Docker + Dev Commands --------

build:
	docker compose -f docker-compose.dev.yml build

shell:
	docker compose -f docker-compose.dev.yml run --rm rlvr bash

up:
	docker compose -f docker-compose.dev.yml up

down:
	docker compose -f docker-compose.dev.yml down

train:
	uv run scripts/train_model.py --config config/train_config.yaml

eval:
	uv run scripts/evaluate_model.py

format:
	ruff check --fix .

test:
	uv run pytest -q

lint:
	ruff check .

	