"""Estatística descritiva com Python puro e fórmulas visíveis."""

from collections.abc import Sequence
from math import sqrt


def _validated(values: Sequence[float]) -> list[float]:
    """Converte uma sequência não vazia em ``float`` e rejeita valores NaN.

    A função privada centraliza pré-condições para que média, variância e desvio-padrão compartilhem
    o mesmo contrato. A verificação ``value != value`` funciona porque NaN é o único número de
    ponto flutuante que não é igual a si próprio.
    """
    if not values:
        raise ValueError("A sequência não pode estar vazia.")
    converted = [float(value) for value in values]
    if any(value != value for value in converted):
        raise ValueError("Valores NaN não são aceitos nesta implementação didática.")
    return converted


def mean(values: Sequence[float]) -> float:
    """Calcula a média aritmética ``Σx_i / n``.

    Args:
        values: Sequência não vazia de números, sem NaN.

    Returns:
        Centro aritmético expresso na mesma unidade dos valores.

    Raises:
        ValueError: Se a sequência estiver vazia ou contiver NaN.
    """
    data = _validated(values)
    return sum(data) / len(data)


def variance(values: Sequence[float], *, sample: bool = False) -> float:
    """Calcula a dispersão quadrática em torno da média.

    Args:
        values: Sequência não vazia de números, sem NaN.
        sample: Quando falso, divide ``Σ(x_i - média)²`` por ``n`` e descreve os próprios valores.
            Quando verdadeiro, divide por ``n - 1`` (correção de Bessel) para estimar a variância
            de uma população a partir de uma amostra.

    Returns:
        Variância populacional ou amostral, na unidade original ao quadrado.

    Raises:
        ValueError: Se os dados forem inválidos ou a variância amostral receber menos de dois
            valores.
    """
    data = _validated(values)
    if sample and len(data) < 2:
        raise ValueError("A variância amostral exige pelo menos dois valores.")
    center = mean(data)
    divisor = len(data) - 1 if sample else len(data)
    return sum((value - center) ** 2 for value in data) / divisor


def standard_deviation(values: Sequence[float], *, sample: bool = False) -> float:
    """Calcula a raiz da variância e recupera a unidade original dos dados.

    O argumento ``sample`` possui exatamente o mesmo significado de :func:`variance`.
    """
    return sqrt(variance(values, sample=sample))
