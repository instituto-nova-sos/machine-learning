"""Operações vetoriais mínimas antes de NumPy."""

from collections.abc import Sequence


def _same_size(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError("Os vetores devem ter o mesmo tamanho.")
    if not left:
        raise ValueError("Os vetores não podem estar vazios.")


def add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    """Soma vetores componente a componente."""
    _same_size(left, right)
    return [float(a) + float(b) for a, b in zip(left, right, strict=True)]


def scalar_multiply(scalar: float, vector: Sequence[float]) -> list[float]:
    """Multiplica cada componente por um escalar."""
    if not vector:
        raise ValueError("O vetor não pode estar vazio.")
    return [float(scalar) * float(value) for value in vector]


def manual_dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Calcula produto escalar; NumPy é preferível para grandes vetores."""
    _same_size(left, right)
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
