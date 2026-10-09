"""Derivadas analíticas, numéricas e atualização são verificações distintas."""

import numpy as np
import pytest

from sos_ml.from_scratch.backpropagation import (
    fit,
    gradient_check,
    gradients,
    parameters,
    scalar_example,
)
from sos_ml.from_scratch.neural_network import BinaryMLP


def test_scalar_chain_by_hand() -> None:
    """z=1,1; a=1,21; dJ/dz=0,462; dJ/dw=0,924."""
    values = scalar_example()
    assert values["perda"] == pytest.approx(0.02205)
    assert values["dJ_dw"] == pytest.approx(0.924)
    assert values["dJ_db"] == pytest.approx(0.462)


def test_every_parameter_and_restore() -> None:
    """Diferenças finitas testam TODOS os pesos/vieses em lote assimétrico."""
    model = BinaryMLP.initialize()
    X, y = [[-0.7, 1.2], [0.5, -0.3], [1.4, 0.8]], [0, 1, 0]
    before = [p.copy() for p in parameters(model)]
    assert gradient_check(model, X, y) < 1e-8
    for a, b in zip(before, parameters(model), strict=True):
        np.testing.assert_array_equal(a, b)
    for p, d in zip(parameters(model), gradients(model, X, y), strict=True):
        assert p.shape == d.shape


def test_training_reduces_loss() -> None:
    """Um lote pequeno separável verifica ajuste sem prometer utilidade externa."""
    model = BinaryMLP.initialize()
    X, y = [[-1, -1], [1, 1]], [0, 1]
    initial = model.loss(X, y)
    history = fit(model, X, y, epochs=100)
    assert history[-1] < initial / 2
    with pytest.raises(ValueError):
        fit(model, X, y, learning_rate=float("nan"))
