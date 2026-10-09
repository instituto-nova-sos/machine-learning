"""MLP binária pequena: entrada → ReLU oculta → logit, sem framework."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .dense_layer import DenseLayer, FloatArray
from .logistic_regression import binary_cross_entropy_from_logits, validate_targets


@dataclass
class BinaryMLP:
    """Duas camadas, com W₁(d,h), b₁(h,), W₂(h,1), b₂(1,).

    A camada final não aplica sigmoid: a perda recebe logits estáveis. A sigmoid
    é aplicada somente para interpretar probabilidade. O modelo não padroniza
    entradas; escalas ajustadas no treino continuam responsabilidade externa.
    """

    hidden: DenseLayer
    output: DenseLayer

    def __post_init__(self) -> None:
        """Exige ReLU oculta, saída linear única e dimensões conectáveis."""
        if (self.hidden.activation != "relu" or self.output.activation != "linear"
                or self.output.weights.shape != (self.hidden.weights.shape[1], 1)):
            raise ValueError("MLP exige ReLU oculta e uma saída linear compatível.")

    @classmethod
    def initialize(cls, width: int = 2, hidden_size: int = 4, seed: int = 23) -> "BinaryMLP":
        """Cria pesos pseudoaleatórios e vieses zero, com gerador NumPy local.

        Semente 23 reproduz a aula. Pesos diferentes quebram simetria entre
        unidades; escala sqrt(2/d) favorece ReLU sem garantir convergência.
        Requer contagens inteiras positivas; não consome gerador aleatório global.
        """
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (width, hidden_size)):
            raise ValueError("Dimensões devem ser inteiros positivos.")
        rng = np.random.default_rng(seed)
        return cls(
            DenseLayer(rng.normal(0, np.sqrt(2 / width), (width, hidden_size)),
                       np.zeros(hidden_size), "relu"),
            DenseLayer(rng.normal(0, np.sqrt(1 / hidden_size), (hidden_size, 1)),
                       np.zeros(1), "linear"),
        )

    def logits(self, inputs: ArrayLike) -> FloatArray:
        """Retorna z (n,1); mantém eixo de saída para evitar broadcasting com alvos."""
        self.__post_init__()
        return self.output.forward(self.hidden.forward(inputs))

    def predict_proba(self, inputs: ArrayLike) -> list[float]:
        """Retorna P(falha sintética) (n,), sem limiar nem garantia de calibração."""
        return [float(p) for p in np.exp(-np.logaddexp(0, -self.logits(inputs)[:, 0]))]

    def loss(self, inputs: ArrayLike, targets: list[int]) -> float:
        """Mede BCE média dos logits, reutilizando a implementação Python pura."""
        scores = self.logits(inputs)[:, 0].tolist()
        validate_targets(targets, len(scores))
        return binary_cross_entropy_from_logits(scores, targets)
