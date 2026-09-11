"""Testes da perda e da regressão linear univariada implementadas manualmente."""

import pytest

from sos_ml.from_scratch.linear_regression import LinearRegression1D
from sos_ml.from_scratch.losses import mean_squared_error


def test_mse() -> None:
    """Confere ``[(3-2)² + (5-3)²] / 2 = 2,5``."""
    assert mean_squared_error([3, 5], [2, 3]) == pytest.approx(2.5)


def test_model_fits_exact_line() -> None:
    """O gradiente deve recuperar ``y = 2x + 1`` quando os dados não contêm ruído."""
    features = [0.0, 1.0, 2.0, 3.0]
    targets = [1.0, 3.0, 5.0, 7.0]
    model = LinearRegression1D()
    history = model.fit(features, targets, learning_rate=0.05, epochs=1000)
    # A perda final menor confirma otimização; peso e viés confirmam a solução esperada.
    assert history[-1] < history[0]
    assert model.weight == pytest.approx(2.0, abs=1e-5)
    assert model.bias == pytest.approx(1.0, abs=1e-5)


def test_invalid_training_inputs() -> None:
    """Cada entrada precisa de exatamente um alvo correspondente."""
    with pytest.raises(ValueError, match="mesmo tamanho"):
        LinearRegression1D().fit([1, 2], [3])
