"""Um neurônio em Python puro compõe produto escalar, viés e ativação."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .activations import relu, sigmoid
from .logistic_regression import validate_features
from .vectors import manual_dot_product

Activation = Literal["linear", "sigmoid", "relu"]


@dataclass
class Neuron:
    """Recebe x (d,) e retorna um escalar; pesos têm a mesma ordem de atributos.

    ``linear`` preserva z; ``sigmoid`` transforma z em [0,1]; ``relu`` retifica.
    Pesos e viés são números explícitos, não um modelo já treinado. Esta classe
    só executa propagação para frente, não ajuste. Use entradas padronizadas ao
    comparar parâmetros com a logística; unidades dependem do contrato de x.
    """

    weights: list[float]
    bias: float = 0.0
    activation: Activation = "linear"

    def forward(self, inputs: Sequence[float]) -> float:
        """Calcula z=Σw_j*x_j+b e a=g(z), rejeitando formas ou valores inválidos.

        Não modifica entradas ou parâmetros. Um neurônio sigmoid corresponde
        exatamente à probabilidade da logística binária, antes de qualquer limiar.
        """
        validate_features([inputs], len(self.weights))
        validate_features([self.weights], len(self.weights))
        score = manual_dot_product(inputs, self.weights) + self.bias
        if not math.isfinite(score):
            raise ValueError("O escore e o viés devem ser finitos.")
        if self.activation == "sigmoid":
            return sigmoid(score)
        if self.activation == "relu":
            return relu(score)
        if self.activation == "linear":
            return score
        raise ValueError("Ativação deve ser linear, sigmoid ou relu.")
