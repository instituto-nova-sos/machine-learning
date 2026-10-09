"""Camada densa NumPy: vários produtos escalares em uma multiplicação matricial."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .neuron import Activation

FloatArray: TypeAlias = NDArray[np.float64]


def matrix(values: ArrayLike, width: int) -> FloatArray:
    """Converte X para float64 (n,d), n>0 e d=width; rejeita NaN e broadcasting errado."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != width:
        raise ValueError("Forneça matriz (exemplos, atributos) com largura compatível.")
    if not np.all(np.isfinite(result)):
        raise ValueError("A matriz deve conter valores finitos.")
    return result


@dataclass
class DenseLayer:
    """Guarda W (d,h), b (h,) e g; cada coluna de W pertence a um neurônio.

    ``forward(X)`` calcula Z=X@W+b e A=g(Z), ambos (n,h). Bias é repetido
    nas linhas por broadcasting intencional. Arrays são copiados na construção
    para não modificar os parâmetros do chamador. Não há autograd nesta classe.
    """

    weights: FloatArray
    bias: FloatArray
    activation: Activation = "relu"

    def __post_init__(self) -> None:
        """Copia parâmetros e valida dimensões, finitude e nome de ativação."""
        self.weights = np.array(self.weights, dtype=np.float64, copy=True)
        self.bias = np.array(self.bias, dtype=np.float64, copy=True)
        if self.weights.ndim != 2 or min(self.weights.shape) < 1:
            raise ValueError("W deve ter forma (entradas, saídas) não vazia.")
        if self.bias.shape != (self.weights.shape[1],):
            raise ValueError("b deve ter um valor por saída, sem eixo de lote.")
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise ValueError("Parâmetros devem ser finitos.")
        if self.activation not in ("linear", "sigmoid", "relu"):
            raise ValueError("Ativação deve ser linear, sigmoid ou relu.")

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Propaga X (n,d) para A (n,h), sem alterar parâmetros ou guardar cache.

        Sigmoid usa logaddexp para evitar exp de positivos extremos. A saída
        linear é adequada a logits; ReLU a representações ocultas. Rejeita
        escores não finitos, inclusive se a multiplicação exceder float64.
        """
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise ValueError("Parâmetros devem ser finitos.")
        X = matrix(inputs, self.weights.shape[0])
        scores = X @ self.weights + self.bias
        if not np.all(np.isfinite(scores)):
            raise ValueError("Escores não finitos; verifique parâmetros e escala.")
        if self.activation == "relu":
            return np.maximum(scores, 0)
        if self.activation == "sigmoid":
            return np.exp(-np.logaddexp(0, -scores))
        return scores
