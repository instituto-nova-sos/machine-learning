"""Testes das fórmulas de estatística descritiva implementadas com Python puro."""

import pytest

from sos_ml.from_scratch.statistics import mean, standard_deviation, variance


def test_descriptive_statistics() -> None:
    """Confere média, variância populacional e desvio-padrão em um exemplo calculável à mão."""
    values = [1.0, 2.0, 3.0]
    assert mean(values) == pytest.approx(2.0)
    assert variance(values) == pytest.approx(2 / 3)
    assert standard_deviation(values) == pytest.approx((2 / 3) ** 0.5)


def test_empty_values_are_rejected() -> None:
    """Uma estatística sem observações deve falhar, pois divisão por ``n = 0`` não é definida."""
    with pytest.raises(ValueError, match="vazia"):
        mean([])
