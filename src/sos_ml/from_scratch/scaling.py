"""Padronização univariada transparente."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class StandardScaler1D:
    """Transforma x em (x-média)/desvio; deve ser ajustado somente no treino."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, values: Sequence[float]) -> "StandardScaler1D":
        if not values:
            raise ValueError("Forneça valores de treino para ajustar a escala.")
        converted = [float(value) for value in values]
        center = sum(converted) / len(converted)
        variance = sum((value - center) ** 2 for value in converted) / len(converted)
        scale = sqrt(variance)
        if scale == 0:
            raise ValueError("Não é possível padronizar um atributo constante.")
        return cls(mean=center, scale=scale)

    def transform(self, values: Sequence[float]) -> list[float]:
        return [(float(value) - self.mean) / self.scale for value in values]
