import pytest

from sos_ml.from_scratch.data_split import train_test_indices
from sos_ml.from_scratch.scaling import StandardScaler1D


def test_split_is_deterministic_and_disjoint() -> None:
    first = train_test_indices(20, seed=7)
    second = train_test_indices(20, seed=7)
    assert first == second
    train, test = first
    assert not set(train).intersection(test)
    assert sorted(train + test) == list(range(20))


def test_standard_scaler() -> None:
    scaler = StandardScaler1D.fit([1, 2, 3])
    transformed = scaler.transform([1, 2, 3])
    assert sum(transformed) / len(transformed) == pytest.approx(0)
    assert sum(value**2 for value in transformed) / len(transformed) == pytest.approx(1)


def test_constant_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="constante"):
        StandardScaler1D.fit([2, 2])
