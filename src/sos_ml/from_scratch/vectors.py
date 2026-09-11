"""Operações vetoriais mínimas, implementadas antes de introduzir NumPy.

Vetores são representados por sequências unidimensionais. As compreensões deixam explícita a
correspondência entre componentes que bibliotecas numéricas depois executarão de forma vetorizada.
"""

from collections.abc import Sequence


def _same_size(left: Sequence[float], right: Sequence[float]) -> None:
    """Valida a precondição comum de duas operações vetoriais binárias."""
    if len(left) != len(right):
        raise ValueError("Os vetores devem ter o mesmo tamanho.")
    if not left:
        raise ValueError("Os vetores não podem estar vazios.")


def add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    """Soma dois vetores componente a componente.

    Para vetores ``a`` e ``b`` de dimensão ``n``, produz ``c_i = a_i + b_i``. A posição de cada
    componente precisa representar o mesmo atributo nos dois operandos.

    Raises:
        ValueError: Se os vetores estiverem vazios ou tiverem dimensões diferentes.
    """
    _same_size(left, right)
    return [float(a) + float(b) for a, b in zip(left, right, strict=True)]


def scalar_multiply(scalar: float, vector: Sequence[float]) -> list[float]:
    """Multiplica cada componente de um vetor pelo mesmo escalar.

    Implementa ``resultado_i = escalar * vetor_i`` e retorna uma nova lista, sem modificar a
    sequência recebida.

    Raises:
        ValueError: Se o vetor estiver vazio.
    """
    if not vector:
        raise ValueError("O vetor não pode estar vazio.")
    return [float(scalar) * float(value) for value in vector]


def manual_dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Calcula o produto escalar ``Σ(a_i * b_i)``.

    O resultado condensa dois vetores em um escalar. Em modelos lineares, um vetor pode representar
    atributos e o outro pesos; cada produto mede a contribuição de um atributo para a soma.

    Raises:
        ValueError: Se os vetores estiverem vazios ou tiverem dimensões diferentes.

    Note:
        A implementação privilegia transparência. NumPy é mais apropriado para vetores grandes.
    """
    _same_size(left, right)
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
