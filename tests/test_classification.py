"""Verificações matemáticas e de contrato do novo classificador, com dados pequenos."""

import math

import numpy as np
import pytest

from sos_ml.from_scratch.activations import probability_to_class, sigmoid
from sos_ml.from_scratch.logistic_regression import (
    LogisticRegressionBinary,
    binary_cross_entropy_from_logits,
)
from sos_ml.from_scratch.logistic_regression_numpy import LogisticRegressionNumpy
from sos_ml.sklearn_models.logistic_regression import fit_logistic_regression


@pytest.mark.parametrize("score, expected", [(-1000, 0), (0, 0.5), (1000, 1)])
def test_sigmoid_extremes(score: float, expected: float) -> None:
    """Escores extremos não podem causar overflow; zero representa incerteza simétrica."""
    assert sigmoid(score) == expected


def test_threshold_changes_only_decision() -> None:
    """O empate é inclusivo, os extremos são válidos e o limiar não altera parâmetros."""
    model = LogisticRegressionBinary([1.0])
    probability = model.predict_proba([[0.0]])
    assert model.predict([[0.0]], threshold=0.5) == [1]
    assert model.predict([[0.0]], threshold=0.6) == [0]
    assert probability_to_class(0.0, threshold=0.0) == 1
    assert probability_to_class(1.0, threshold=1.0) == 1
    assert model.predict_proba([[0.0]]) == probability


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_activation_input(value: float) -> None:
    """NaN e infinitos não representam medidas válidas para a aula."""
    with pytest.raises(ValueError):
        sigmoid(value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_or_threshold(value: float) -> None:
    """Ambos os argumentos precisam respeitar o intervalo de probabilidades."""
    with pytest.raises(ValueError):
        probability_to_class(value)
    with pytest.raises(ValueError):
        probability_to_class(0.5, threshold=value)


def test_stable_cross_entropy() -> None:
    """Com p=0,5 a perda é log(2); erro confiante deve continuar sendo penalizado."""
    assert binary_cross_entropy_from_logits([0, 0], [0, 1]) == pytest.approx(math.log(2))
    assert binary_cross_entropy_from_logits([-1000, 1000], [1, 0]) == pytest.approx(1000)
    assert binary_cross_entropy_from_logits([-1000, 1000], [0, 1]) == pytest.approx(0)
    with pytest.raises(ValueError):
        binary_cross_entropy_from_logits([float("nan")], [1])


@pytest.mark.parametrize("model_type", [LogisticRegressionBinary, LogisticRegressionNumpy])
def test_first_step_by_hand(model_type: type[LogisticRegressionBinary]) -> None:
    """x=[-1,1], y=[0,1]: dw=-0,5 e db=0; taxa 0,1 produz w=0,05."""
    model = model_type([0.0])
    dw, db = model.gradients([[-1.0], [1.0]], [0, 1])
    assert dw == pytest.approx([-0.5])
    assert db == pytest.approx(0)
    history = model.fit([[-1.0], [1.0]], [0, 1], epochs=1)
    assert model.weights == pytest.approx([0.05])
    assert model.bias == pytest.approx(0)
    assert history == pytest.approx([model.loss([[-1.0], [1.0]], [0, 1])])
    assert history[0] < math.log(2)


@pytest.mark.parametrize("model_type", [LogisticRegressionBinary, LogisticRegressionNumpy])
def test_all_gradients_by_finite_differences(model_type: type[LogisticRegressionBinary]) -> None:
    """Compara cada peso e o viés a diferenças finitas em dados assimétricos."""
    X = [[-0.7, 1.2], [0.5, -0.3], [1.4, 0.8]]
    y = [0, 1, 0]
    model = model_type([0.3, -0.2], 0.1)
    dw, db = model.gradients(X, y)
    epsilon = 1e-6
    for index in range(2):
        original = model.weights[index]
        model.weights[index] = original + epsilon
        plus = model.loss(X, y)
        model.weights[index] = original - epsilon
        minus = model.loss(X, y)
        model.weights[index] = original
        assert dw[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-8)
    model.bias = 0.1 + epsilon
    plus = model.loss(X, y)
    model.bias = 0.1 - epsilon
    minus = model.loss(X, y)
    assert db == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-8)


def test_three_implementations_on_nonseparable_data() -> None:
    """Rótulos sobrepostos dão ótimo finito e permitem comparar diferentes otimizadores.

    Em cada x=-1, 1/4 dos alvos é positivo; em x=1, 3/4 são positivos. A solução
    analítica é w=log(3), b=0, portanto as probabilidades são 0,25 e 0,75.
    """
    X = [[-1.0]] * 4 + [[1.0]] * 4
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    manual = LogisticRegressionBinary([0.0])
    vectorized = LogisticRegressionNumpy([0.0])
    history = manual.fit(X, y, epochs=1000)
    vector_history = vectorized.fit(X, y, epochs=1000)
    professional = fit_logistic_regression(np.asarray(X), np.asarray(y, dtype=np.int64))
    assert manual.weights == pytest.approx([math.log(3)], abs=1e-6)
    assert manual.predict_proba([[-1.0], [1.0]]) == pytest.approx([0.25, 0.75], abs=1e-6)
    assert vector_history == pytest.approx(history, abs=1e-12)
    assert professional.predict_proba(X)[:, 1] == pytest.approx(manual.predict_proba(X), abs=1e-6)


@pytest.mark.parametrize("model_type", [LogisticRegressionBinary, LogisticRegressionNumpy])
@pytest.mark.parametrize(
    "features, targets",
    [
        ([], []),
        ([[]], [0]),
        ([[1, 2], [1]], [0, 1]),
        ([[1, 2]], []),
        ([[1, float("nan")]], [0]),
        ([[1, float("inf")]], [0]),
        ([[1, 2]], [2]),
        ([[1, 2], [3, 4]], [0, 0]),
    ],
)
def test_invalid_training_data(
    model_type: type[LogisticRegressionBinary], features: list[list[float]], targets: list[int]
) -> None:
    """Formas, finitude, rótulos e presença das duas classes são pré-condições explícitas."""
    with pytest.raises(ValueError):
        model_type([0.0, 0.0]).fit(features, targets)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_learning_rate(rate: float) -> None:
    """Uma taxa inválida deve falhar antes da primeira atualização."""
    with pytest.raises(ValueError):
        LogisticRegressionBinary([0.0]).fit([[-1], [1]], [0, 1], learning_rate=rate)


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5])
def test_invalid_epochs(epochs: int) -> None:
    """Não aceitamos booleanos ou frações como quantidade de épocas."""
    with pytest.raises(ValueError):
        LogisticRegressionBinary([0.0]).fit([[-1], [1]], [0, 1], epochs=epochs)


@pytest.mark.parametrize("model_type", [LogisticRegressionBinary, LogisticRegressionNumpy])
def test_extreme_model_scores(model_type: type[LogisticRegressionBinary]) -> None:
    """As APIs completas também preservam perda finita nos extremos da sigmoid."""
    model = model_type([1000.0])
    assert model.predict_proba([[-1.0], [1.0]]) == [0.0, 1.0]
    assert model.loss([[-1.0], [1.0]], [1, 0]) == pytest.approx(1000.0)


@pytest.mark.parametrize("weights, bias", [([], 0), ([float("nan")], 0), ([0], float("inf"))])
def test_invalid_initial_parameters(weights: list[float], bias: float) -> None:
    """Recusa estado inicial sem interpretação numérica válida."""
    with pytest.raises(ValueError):
        LogisticRegressionBinary(weights, bias)
