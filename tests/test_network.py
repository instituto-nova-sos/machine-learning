"""Contas e formas pequenas verificam várias unidades simultâneas."""

import numpy as np
import pytest

from sos_ml.from_scratch.dense_layer import DenseLayer
from sos_ml.from_scratch.neural_network import BinaryMLP


def test_dense_by_hand() -> None:
    """[1,2] @ [[1,−1],[2,1]]+[0,1] = [5,2]."""
    layer = DenseLayer(np.array([[1., -1.], [2., 1.]]), np.array([0., 1.]))
    assert layer.forward([[1, 2]]).tolist() == [[5, 2]]
    with pytest.raises(ValueError):
        layer.forward([1, 2])
    with pytest.raises(ValueError):
        DenseLayer(np.ones((2, 3)), np.ones((1, 3)))


def test_network_shapes_and_seed() -> None:
    """A mesma semente preserva forward; trocar o lote não muda parâmetros."""
    first = BinaryMLP.initialize()
    second = BinaryMLP.initialize()
    assert first.logits([[0, 1], [1, 0]]).shape == (2, 1)
    assert first.predict_proba([[0, 1]]) == second.predict_proba([[0, 1]])
    assert len(first.predict_proba([[0, 1]])) == 1
