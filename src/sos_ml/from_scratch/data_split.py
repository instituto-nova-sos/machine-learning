"""Partição determinística de índices sem depender de framework de ML."""

import random


def train_test_indices(
    size: int, *, test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Retorna índices disjuntos de treino e teste."""
    if size < 2:
        raise ValueError("A partição exige pelo menos dois exemplos.")
    if not 0 < test_fraction < 1:
        raise ValueError("A fração de teste deve estar entre 0 e 1.")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    test_size = max(1, round(size * test_fraction))
    if test_size >= size:
        test_size = size - 1
    return indices[test_size:], indices[:test_size]
