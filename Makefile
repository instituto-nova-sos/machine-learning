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

.PHONY: run-foundations run-local-ai validate-torch

run-foundations:
	python -m sos_ml.assess
	python -m sos_ml.generalization --plot artifacts/generalizacao.png
	python -m sos_ml.network_demo
	python scripts/jev_opcional.py

run-local-ai:
	python -m sos_ml.local_ai treinar --plot artifacts/mlp_treino_validacao.png
	python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --high-consequence --measure

# A base segue sem exigir torch; este alvo falha cedo se a etapa não foi instalada.
validate-torch:
	python -c "import torch; print('PyTorch instalado:', torch.__version__)"
	python -m pytest tests/test_torch_foundations.py
	python -m sos_ml.torch_demo treinar
	python -m sos_ml.torch_demo inferir

validate: run-foundations run-local-ai
