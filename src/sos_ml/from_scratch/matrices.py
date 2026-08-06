"""Operações matriciais pequenas e explícitas."""

from collections.abc import Sequence

from .vectors import manual_dot_product


def _shape(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("A matriz não pode estar vazia.")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("Todas as linhas devem ter o mesmo tamanho.")
    return len(matrix), columns


def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Troca linhas por colunas."""
    rows, columns = _shape(matrix)
    return [[float(matrix[i][j]) for i in range(rows)] for j in range(columns)]


def matrix_vector_multiply(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    """Calcula A @ x validando dimensões."""
    _, columns = _shape(matrix)
    if len(vector) != columns:
        raise ValueError("O vetor deve ter uma entrada para cada coluna da matriz.")
    return [manual_dot_product(row, vector) for row in matrix]


def matrix_multiply(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Calcula produto matricial para matrizes pequenas."""
    _, left_columns = _shape(left)
    right_rows, _ = _shape(right)
    if left_columns != right_rows:
        raise ValueError("Dimensões internas incompatíveis para multiplicação.")
    right_t = transpose(right)
    return [[manual_dot_product(row, column) for column in right_t] for row in left]
