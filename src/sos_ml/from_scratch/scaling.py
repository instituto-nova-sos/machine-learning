"""Padronização univariada com parâmetros aprendidos exclusivamente no treino."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class StandardScaler1D:
    """Armazena média e escala para transformar ``x`` em ``(x - média) / escala``.

    Ajustar esses parâmetros somente no conjunto de treino impede que estatísticas do teste vazem
    para o processo de desenvolvimento. A classe é imutável para evitar mudanças acidentais entre
    treino, avaliação e inferência.

    Attributes:
        mean: Média populacional dos valores usados no ajuste.
        scale: Desvio-padrão populacional desses valores, obrigatoriamente diferente de zero.
    """

    mean: float
    scale: float

    @classmethod
    def fit(cls, values: Sequence[float]) -> "StandardScaler1D":
        """Calcula média e desvio-padrão populacional de valores de treino.

        Raises:
            ValueError: Se a sequência estiver vazia ou representar um atributo constante.
        """
        if not values:
            raise ValueError("Forneça valores de treino para ajustar a escala.")
        converted = [float(value) for value in values]
        center = sum(converted) / len(converted)
        # Usamos divisor ``n`` porque descrevemos o conjunto usado para ajustar a transformação;
        # não estamos estimando aqui a variância de uma população por inferência estatística.
        variance = sum((value - center) ** 2 for value in converted) / len(converted)
        scale = sqrt(variance)
        if scale == 0:
            raise ValueError("Não é possível padronizar um atributo constante.")
        return cls(mean=center, scale=scale)

    def transform(self, values: Sequence[float]) -> list[float]:
        """Aplica os parâmetros já ajustados sem recalculá-los nos novos dados.

        Valores de treino transformados tendem a ter média zero e variância um. Dados de teste ou
        produção não precisam preservar exatamente essas propriedades, pois vêm de outra amostra.
        """
        return [(float(value) - self.mean) / self.scale for value in values]
