"""Gradiente descendente para funções escalares."""

from collections.abc import Callable


def minimize_scalar(
    derivative: Callable[[float], float],
    initial_value: float,
    *,
    learning_rate: float,
    iterations: int,
) -> list[float]:
    """Produz o caminho x <- x - taxa * f'(x)."""
    if learning_rate <= 0 or iterations <= 0:
        raise ValueError("Taxa de aprendizado e iterações devem ser positivas.")
    path = [float(initial_value)]
    for _ in range(iterations):
        path.append(path[-1] - learning_rate * derivative(path[-1]))
    return path


def finite_difference(
    function: Callable[[float], float], value: float, epsilon: float = 1e-6
) -> float:
    """Aproxima a derivada por diferença central; útil para checagem, não para treino grande."""
    if epsilon <= 0:
        raise ValueError("Epsilon deve ser positivo.")
    return (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)
