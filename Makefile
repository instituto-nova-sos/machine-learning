.PHONY: setup test lint format typecheck run-example run-classification validate

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

format:
	python -m ruff format .

typecheck:
	python -m mypy

run-example:
	python scripts/gerar_dados_imoveis.py
	python -m sos_ml.train --data data/processed/imoveis.csv --output artifacts/modelo_linear.json
	python -m sos_ml.evaluate --data data/processed/imoveis.csv --model artifacts/modelo_linear.json
	python -m sos_ml.predict --model artifacts/modelo_linear.json --area 85

run-classification:
	python -m sos_ml.classify --plot artifacts/classificacao_e_perda.png

validate: lint typecheck test run-example run-classification
