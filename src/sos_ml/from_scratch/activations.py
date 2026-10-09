"""Ativações escalares: a ponte entre um escore real e uma probabilidade binária."""

import math


def sigmoid(score: float) -> float:
    """Calcula ``1 / (1 + exp(-z))`` para um escore finito ``z``.

    O escore não tem unidade; pode ser negativo e não é uma probabilidade. A sigmoid é
    crescente, vale 0,5 em zero e se aproxima de 0 e 1 nos extremos. Por arredondamento,
    o computador pode devolver exatamente esses extremos. Por exemplo, ``sigmoid(0) == 0.5``.

    Usamos duas expressões equivalentes para nunca exponenciar um número positivo grande:
    ``exp(-1000)`` pode arredondar para zero, mas ``exp(1000)`` causaria overflow.
    Levanta ValueError para NaN ou infinito; não modifica nenhum estado.
    """
    if not math.isfinite(score):
        raise ValueError("O escore deve ser finito.")
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def probability_to_class(probability: float, *, threshold: float = 0.5) -> int:
    """Converte probabilidade em classe: retorna 1 se ``p >= limiar``, senão 0.

    Probabilidade e limiar devem ser finitos no intervalo fechado [0, 1]; valores
    inválidos levantam ValueError. O empate pertence à classe 1 por convenção explícita.
    Com limiar zero todos recebem alerta; com limiar um só recebe alerta uma saída igual
    a um. Mudar o limiar não treina novamente nem altera a probabilidade do modelo.
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("A probabilidade deve estar entre 0 e 1 e ser finita.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("O limiar deve estar entre 0 e 1 e ser finito.")
    return int(probability >= threshold)


def relu(score: float) -> float:
    """Retorna max(0,z), uma ativação não linear que não é probabilidade.

    ReLU significa unidade linear retificada (rectified linear unit). Valores
    negativos viram zero; positivos passam sem limite superior. Em zero não há
    derivada única; no backprop adotaremos a convenção zero. Rejeita não finitos.
    """
    if not math.isfinite(score):
        raise ValueError("O escore deve ser finito.")
    return max(0.0, score)
