"""Operações matriciais pequenas com formas e laços conceitualmente explícitos.

Uma matriz é uma sequência de linhas, e cada linha é uma sequência numérica. Essa representação
didática permite enxergar índices e produtos escalares antes de usar arrays do NumPy.
"""

from collections.abc import Sequence

from .vectors import manual_dot_product


def _shape(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    """Valida uma matriz retangular não vazia e retorna ``(linhas, colunas)``."""
    if not matrix or not matrix[0]:
        raise ValueError("A matriz não pode estar vazia.")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("Todas as linhas devem ter o mesmo tamanho.")
    return len(matrix), columns


def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Transpõe uma matriz de forma ``(m, n)`` para ``(n, m)``.

    O elemento de posição ``[i][j]`` na entrada ocupa ``[j][i]`` na saída. Uma nova estrutura é
    criada; a matriz recebida não é alterada.
    """
    rows, columns = _shape(matrix)
    return [[float(matrix[i][j]) for i in range(rows)] for j in range(columns)]


def matrix_vector_multiply(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    """Calcula o produto ``A @ x`` entre uma matriz e um vetor.

    Se ``A`` possui forma ``(m, n)``, ``x`` deve conter ``n`` componentes e o resultado terá ``m``
    valores. Cada saída é o produto escalar entre uma linha de ``A`` e ``x``.

    Raises:
        ValueError: Se a matriz for inválida ou o comprimento do vetor não coincidir com o número
            de colunas.
    """
    _, columns = _shape(matrix)
    if len(vector) != columns:
        raise ValueError("O vetor deve ter uma entrada para cada coluna da matriz.")
    return [manual_dot_product(row, vector) for row in matrix]


def matrix_multiply(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Multiplica matrizes por produtos escalares entre linhas e colunas.

    Para ``A`` de forma ``(m, n)`` e ``B`` de forma ``(n, p)``, retorna ``C`` de forma ``(m, p)``,
    onde ``C[i][j]`` é o produto escalar da linha ``i`` de ``A`` com a coluna ``j`` de ``B``.

    Raises:
        ValueError: Se alguma matriz for vazia ou irregular, ou se as dimensões internas forem
            incompatíveis.
    """
    _, left_columns = _shape(left)
    right_rows, _ = _shape(right)
    if left_columns != right_rows:
        raise ValueError("Dimensões internas incompatíveis para multiplicação.")
    # Transpor uma vez transforma colunas em sequências diretamente iteráveis e evita reconstruir
    # cada coluna dentro do laço de produtos escalares.
    right_t = transpose(right)
    return [[manual_dot_product(row, column) for column in right_t] for row in left]
