import pytest

from sos_ml.from_scratch.linear_regression import LinearRegression1D
from sos_ml.from_scratch.losses import mean_squared_error


def test_mse() -> None:
    assert mean_squared_error([3, 5], [2, 3]) == pytest.approx(2.5)


def test_model_fits_exact_line() -> None:
    features = [0.0, 1.0, 2.0, 3.0]
    targets = [1.0, 3.0, 5.0, 7.0]
    model = LinearRegression1D()
    history = model.fit(features, targets, learning_rate=0.05, epochs=1000)
    assert history[-1] < history[0]
    assert model.weight == pytest.approx(2.0, abs=1e-5)
    assert model.bias == pytest.approx(1.0, abs=1e-5)


def test_invalid_training_inputs() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        LinearRegression1D().fit([1, 2], [3])
