"""Testes do contrato tabular, da reprodutibilidade e da persistência do modelo."""

from pathlib import Path

import pandas as pd
import pytest

from sos_ml.data import generate_housing_data, load_housing_data
from sos_ml.from_scratch.linear_regression import LinearRegression1D
from sos_ml.model_io import load_model, save_model


def test_data_generation_is_deterministic() -> None:
    """Duas gerações com tamanho e semente iguais devem produzir tabelas idênticas."""
    first = generate_housing_data(size=20, seed=7)
    second = generate_housing_data(size=20, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_data_schema_validation(tmp_path: Path) -> None:
    """Um CSV sem as colunas do contrato deve ser rejeitado antes do treinamento."""
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"outra": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="colunas"):
        load_housing_data(path)


def test_model_round_trip(tmp_path: Path) -> None:
    """Salvar e carregar deve preservar a função de previsão, não apenas criar um arquivo."""
    path = tmp_path / "model.json"
    original = LinearRegression1D(weight=2.5, bias=10.0)
    save_model(original, path)
    restored = load_model(path)
    # ``approx`` expressa que parâmetros serializados são números de ponto flutuante.
    assert restored.predict_one(4) == pytest.approx(original.predict_one(4))
