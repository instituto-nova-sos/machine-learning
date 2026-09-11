"""Otimização escalar e verificação numérica de derivadas com Python puro."""

from collections.abc import Callable


def minimize_scalar(
    derivative: Callable[[float], float],
    initial_value: float,
    *,
    learning_rate: float,
    iterations: int,
) -> list[float]:
    """Executa gradiente descendente e devolve todo o caminho percorrido.

    Em cada iteração aplica ``x_novo = x_atual - taxa * derivada(x_atual)``. Subtrair a derivada
    move ``x`` contra o crescimento local da função que se deseja minimizar.

    Args:
        derivative: Função que recebe o valor atual e devolve a derivada da função objetivo nele.
        initial_value: Ponto inicial da busca.
        learning_rate: Tamanho multiplicativo de cada passo; deve ser positivo.
        iterations: Número de atualizações; deve ser positivo.

    Returns:
        Lista com ``iterations + 1`` valores. O primeiro é o ponto inicial e cada item seguinte é
        o resultado de uma atualização, permitindo estudar convergência ou divergência.

    Raises:
        ValueError: Se a taxa ou a quantidade de iterações não for positiva.
    """
    if learning_rate <= 0 or iterations <= 0:
        raise ValueError("Taxa de aprendizado e iterações devem ser positivas.")
    path = [float(initial_value)]
    for _ in range(iterations):
        # Consultamos a derivada no último ponto calculado, nunca no ponto inicial fixo.
        path.append(path[-1] - learning_rate * derivative(path[-1]))
    return path


def finite_difference(
    function: Callable[[float], float], value: float, epsilon: float = 1e-6
) -> float:
    """Aproxima ``f'(value)`` pela diferença central.

    A fórmula ``[f(x + ε) - f(x - ε)] / (2ε)`` estima a inclinação usando pontos dos dois lados de
    ``x`` e costuma ser mais precisa que a diferença para a frente com o mesmo ``epsilon``.

    Args:
        function: Função escalar cuja derivada será aproximada.
        value: Ponto em que a derivada é estimada.
        epsilon: Pequena perturbação positiva aplicada nos dois sentidos.

    Returns:
        Aproximação numérica da derivada. Ela é útil para conferir uma fórmula analítica, não para
        substituir backpropagation ou treinar modelos grandes.

    Raises:
        ValueError: Se ``epsilon`` não for positivo.

    Note:
        Um epsilon grande causa erro de aproximação; um epsilon excessivamente pequeno amplifica
        cancelamento e arredondamento de ponto flutuante.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon deve ser positivo.")
    return (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)
