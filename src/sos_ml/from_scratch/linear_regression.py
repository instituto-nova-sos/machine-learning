"""Regressão linear univariada implementada sem framework de ML."""

from collections.abc import Sequence
from dataclasses import dataclass

from .losses import mean_squared_error


@dataclass
class LinearRegression1D:
    """Modelo ŷ = peso * x + viés, treinado por gradiente descendente em lote."""

    weight: float = 0.0
    bias: float = 0.0

    def predict_one(self, feature: float) -> float:
        return self.weight * float(feature) + self.bias

    def predict(self, features: Sequence[float]) -> list[float]:
        if not features:
            raise ValueError("Forneça pelo menos uma entrada para previsão.")
        return [self.predict_one(feature) for feature in features]

    def gradients(
        self, features: Sequence[float], targets: Sequence[float]
    ) -> tuple[float, float]:
        """Calcula derivadas do MSE em relação ao peso e ao viés."""
        if not features or len(features) != len(targets):
            raise ValueError("Entradas e alvos devem ter o mesmo tamanho não nulo.")
        predictions = self.predict(features)
        count = len(features)
        weight_gradient = (2 / count) * sum(
            (prediction - target) * feature
            for feature, target, prediction in zip(features, targets, predictions, strict=True)
        )
        bias_gradient = (2 / count) * sum(
            prediction - target
            for target, prediction in zip(targets, predictions, strict=True)
        )
        return weight_gradient, bias_gradient

    def fit(
        self,
        features: Sequence[float],
        targets: Sequence[float],
        *,
        learning_rate: float = 0.01,
        epochs: int = 1000,
    ) -> list[float]:
        """Ajusta parâmetros e retorna o histórico de MSE."""
        if learning_rate <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")
        if epochs <= 0:
            raise ValueError("O número de épocas deve ser positivo.")
        history: list[float] = []
        for _ in range(epochs):
            weight_gradient, bias_gradient = self.gradients(features, targets)
            self.weight -= learning_rate * weight_gradient
            self.bias -= learning_rate * bias_gradient
            history.append(mean_squared_error(targets, self.predict(features)))
        return history
