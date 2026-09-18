"""As mesmas equações da regressão logística, agora escritas como operações matriciais.

Leia primeiro logistic_regression.py. Esta subclasse reaproveita o ciclo de treino e
o contrato público; substitui escores, gradientes e perda por operações NumPy. Para
comparar as APIs diretamente, aceita listas e devolve listas, convertendo arrays em
cada chamada. Isso privilegia clareza e equivalência, não desempenho de produção.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .logistic_regression import LogisticRegressionBinary, validate_features, validate_targets


class LogisticRegressionNumpy(LogisticRegressionBinary):
    """Versão vetorizada com a mesma inicialização, limiar e treino da classe manual."""

    def decision_function(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Calcula ``X @ w + b``: (n, d) @ (d,) resulta em (n,).

        O escalar b é somado a cada linha por broadcasting. Validamos antes para
        impedir que uma dimensão incorreta produza um cálculo de outro significado.
        """
        self._validate_parameters()
        validate_features(features, len(self.weights))
        matrix: NDArray[np.float64] = np.asarray(features, dtype=np.float64)
        scores = matrix @ np.asarray(self.weights) + self.bias
        if not np.all(np.isfinite(scores)):
            raise ValueError("Os escores devem ser finitos; padronize as entradas.")
        return [float(z) for z in scores]

    def predict_proba(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Aplica sigmoid vetorizada sem calcular exponenciais positivas grandes.

        ``exp(-logaddexp(0, -z))`` equivale a 1/(1+exp(-z)); logaddexp calcula
        o logaritmo da soma de exponenciais de modo numericamente estável.
        """
        scores = np.asarray(self.decision_function(features))
        return [float(p) for p in np.exp(-np.logaddexp(0.0, -scores))]

    def loss(self, features: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
        """Média da perda por linha, preservando estabilidade para escores extremos."""
        scores = np.asarray(self.decision_function(features))
        validate_targets(targets, len(features))
        labels = np.asarray(targets)
        losses = np.maximum(scores, 0) - labels * scores + np.logaddexp(0, -np.abs(scores))
        return float(np.sum(losses / len(scores)))

    def gradients(
        self, features: Sequence[Sequence[float]], targets: Sequence[int]
    ) -> tuple[list[float], float]:
        """Calcula ``X.T @ (p-y) / n`` e ``média(p-y)``.

        X.T tem forma (d, n); multiplicá-la pelo vetor de n resíduos acumula uma
        derivada para cada um dos d atributos. A divisão por n corresponde à perda
        média e mantém a escala do gradiente comparável entre tamanhos de lote.
        """
        probabilities = np.asarray(self.predict_proba(features))
        validate_targets(targets, len(features))
        errors = probabilities - np.asarray(targets)
        gradients = np.asarray(features, dtype=np.float64).T @ (errors / len(features))
        return [float(value) for value in gradients], float(np.mean(errors))
