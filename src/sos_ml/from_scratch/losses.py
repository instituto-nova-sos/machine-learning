"""Funções de perda explícitas para quantificar erros de previsão."""

from collections.abc import Sequence


def mean_squared_error(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Calcula o erro quadrático médio entre alvos e previsões.

    Para ``n`` pares, implementa ``MSE = (1/n) * Σ(y_i - y_hat_i)²``. Elevar ao quadrado impede
    cancelamento entre erros positivos e negativos e atribui peso maior a resíduos grandes.

    Args:
        targets: Valores observados ``y``.
        predictions: Valores estimados ``y_hat`` na mesma ordem dos alvos.

    Returns:
        Média não negativa dos resíduos ao quadrado. A unidade também fica ao quadrado: se o alvo
        está em reais, o MSE está em reais ao quadrado.

    Raises:
        ValueError: Se as sequências estiverem vazias ou tiverem tamanhos diferentes.

    Note:
        Um MSE isolado não demonstra generalização, causalidade, justiça ou utilidade do modelo.
    """
    if not targets or len(targets) != len(predictions):
        raise ValueError("Alvos e previsões devem ter o mesmo tamanho não nulo.")
    return sum(
        (float(target) - float(prediction)) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)
