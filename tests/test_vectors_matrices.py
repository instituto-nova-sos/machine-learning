"""Testes das operações vetoriais e matriciais anteriores à introdução do NumPy."""

import pytest

from sos_ml.from_scratch.matrices import matrix_multiply, matrix_vector_multiply, transpose
from sos_ml.from_scratch.vectors import add, manual_dot_product


def test_vector_operations() -> None:
    """Verifica soma por componente e produto escalar com resultados calculados manualmente."""
    assert add([1, 2], [3, 4]) == [4, 6]
    assert manual_dot_product([2, 3], [4, 5]) == pytest.approx(23)


def test_matrix_operations() -> None:
    """Confere formas e valores de transposição e dos dois tipos de produto matricial."""
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert matrix_vector_multiply([[1, 2], [3, 4]], [2, 1]) == [4, 10]
    assert matrix_multiply([[1, 2]], [[3], [4]]) == [[11]]


def test_invalid_shapes_are_rejected() -> None:
    """Dimensões incompatíveis devem gerar erro em vez de truncamento ou resultado ambíguo."""
    with pytest.raises(ValueError, match="mesmo tamanho"):
        manual_dot_product([1], [1, 2])
    with pytest.raises(ValueError, match="Dimensões internas"):
        matrix_multiply([[1, 2]], [[1, 2]])
