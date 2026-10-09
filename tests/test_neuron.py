"""A unidade artificial recompõe contas já verificadas na regressão logística."""

import pytest

from sos_ml.from_scratch.activations import relu
from sos_ml.from_scratch.logistic_regression import LogisticRegressionBinary
from sos_ml.from_scratch.neuron import Neuron


def test_neuron_by_hand_and_logistic_equivalence() -> None:
    """x=[1,0,5], w=[0,8,0,4], b=-0,2: z=0,8."""
    assert Neuron([0.8, 0.4], -0.2).forward([1, 0.5]) == pytest.approx(0.8)
    unit = Neuron([0.8, 0.4], -0.2, "sigmoid")
    model = LogisticRegressionBinary([0.8, 0.4], -0.2)
    assert unit.forward([1, 0.5]) == model.predict_proba([[1, 0.5]])[0]
    assert [relu(x) for x in (-2, 0, 3)] == [0, 0, 3]


@pytest.mark.parametrize("inputs", [[], [1], [1, float("nan")]])
def test_invalid_neuron_inputs(inputs: list[float]) -> None:
    """Um produto de significado errado deve falhar explicitamente."""
    with pytest.raises(ValueError):
        Neuron([1, 2]).forward(inputs)
