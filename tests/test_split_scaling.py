"""Testes de partição reprodutível e padronização ajustada nos dados fornecidos."""

import pytest

from sos_ml.from_scratch.data_split import train_test_indices
from sos_ml.from_scratch.scaling import StandardScaler1D


def test_split_is_deterministic_and_disjoint() -> None:
    """A mesma semente repete a divisão, sem sobreposição nem perda de índices."""
    first = train_test_indices(20, seed=7)
    second = train_test_indices(20, seed=7)
    assert first == second
    train, test = first
    # Um exemplo em ambos os conjuntos contaminaria a avaliação; um índice ausente descartaria dado.
    assert not set(train).intersection(test)
    assert sorted(train + test) == list(range(20))


def test_standard_scaler() -> None:
    """Dados usados no ajuste devem terminar com média zero e variância populacional um."""
    scaler = StandardScaler1D.fit([1, 2, 3])
    transformed = scaler.transform([1, 2, 3])
    assert sum(transformed) / len(transformed) == pytest.approx(0)
    assert sum(value**2 for value in transformed) / len(transformed) == pytest.approx(1)


def test_constant_feature_is_rejected() -> None:
    """Um atributo sem variação teria escala zero e provocaria divisão por zero."""
    with pytest.raises(ValueError, match="constante"):
        StandardScaler1D.fit([2, 2])
