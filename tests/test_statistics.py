import pytest

from sos_ml.from_scratch.statistics import mean, standard_deviation, variance


def test_descriptive_statistics() -> None:
    values = [1.0, 2.0, 3.0]
    assert mean(values) == pytest.approx(2.0)
    assert variance(values) == pytest.approx(2 / 3)
    assert standard_deviation(values) == pytest.approx((2 / 3) ** 0.5)


def test_empty_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="vazia"):
        mean([])
