"""Funções de perda educacionais."""

from collections.abc import Sequence


def mean_squared_error(targets: Sequence[float], predictions: Sequence[float]) -> float:
    """Retorna a média dos resíduos quadráticos."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("Alvos e previsões devem ter o mesmo tamanho não nulo.")
    return sum(
        (float(target) - float(prediction)) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)
