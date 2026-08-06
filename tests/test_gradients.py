import pytest

from sos_ml.from_scratch.gradient_descent import finite_difference, minimize_scalar
from sos_ml.from_scratch.linear_regression import LinearRegression1D
from sos_ml.from_scratch.losses import mean_squared_error


def test_finite_difference_matches_analytical_derivative() -> None:
    assert finite_difference(lambda x: x**2, 3.0) == pytest.approx(6.0, rel=1e-6)


def test_regression_weight_gradient() -> None:
    features = [1.0, 2.0, 3.0]
    targets = [2.0, 4.0, 6.0]
    model = LinearRegression1D(weight=0.5, bias=0.2)
    analytical, _ = model.gradients(features, targets)

    def loss_at_weight(weight: float) -> float:
        candidate = LinearRegression1D(weight=weight, bias=model.bias)
        return mean_squared_error(targets, candidate.predict(features))

    numerical = finite_difference(loss_at_weight, model.weight)
    assert analytical == pytest.approx(numerical, rel=1e-6)


def test_gradient_descent_approaches_minimum() -> None:
    path = minimize_scalar(lambda x: 2 * (x - 3), 10, learning_rate=0.1, iterations=100)
    assert path[-1] == pytest.approx(3.0, abs=1e-8)
