"""Estatística descritiva com Python puro."""

from collections.abc import Sequence
from math import sqrt


def _validated(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("A sequência não pode estar vazia.")
    converted = [float(value) for value in values]
    if any(value != value for value in converted):
        raise ValueError("Valores NaN não são aceitos nesta implementação didática.")
    return converted


def mean(values: Sequence[float]) -> float:
    """Calcula a média aritmética."""
    data = _validated(values)
    return sum(data) / len(data)


def variance(values: Sequence[float], *, sample: bool = False) -> float:
    """Calcula variância populacional ou amostral (divisor n-1)."""
    data = _validated(values)
    if sample and len(data) < 2:
        raise ValueError("A variância amostral exige pelo menos dois valores.")
    center = mean(data)
    divisor = len(data) - 1 if sample else len(data)
    return sum((value - center) ** 2 for value in data) / divisor


def standard_deviation(values: Sequence[float], *, sample: bool = False) -> float:
    """Calcula o desvio-padrão na mesma convenção de ``variance``."""
    return sqrt(variance(values, sample=sample))
