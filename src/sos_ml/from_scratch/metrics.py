"""Métricas binárias em Python puro: conte erros antes de resumir proporções.

Classe positiva 1 significa falha sintética; linhas da matriz são alvos reais e
colunas são previsões, na ordem [0, 1]. Nenhuma função escolhe limiar ou política.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .logistic_regression import validate_targets


@dataclass(frozen=True)
class ConfusionCounts:
    """Contagens não negativas VP, VN, FP e FN, sem unidade física.

    Exemplo: TN=3, FP=1, FN=2, TP=4 produz matriz [[3,1],[2,4]].
    A construção direta aceita somente inteiros não negativos, com total positivo.
    Métricas indefinidas retornam None: ausência de denominador não é desempenho zero.
    """

    tn: int
    fp: int
    fn: int
    tp: int

    def __post_init__(self) -> None:
        """Rejeita contagens inválidas antes que produzam taxas aparentemente válidas."""
        values = (self.tn, self.fp, self.fn, self.tp)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError("As contagens devem ser inteiros não negativos.")
        if sum(values) == 0:
            raise ValueError("A avaliação exige pelo menos um exemplo.")

    @property
    def matrix(self) -> list[list[int]]:
        """Retorna [[VN, FP], [FN, VP]], alvo nas linhas e previsão nas colunas."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def metrics(self) -> dict[str, float | None]:
        """Calcula taxas em [0,1]; None indica razão com denominador zero.

        Precisão pergunta quantos alertas são positivos; revocação pergunta quantos
        positivos foram detectados. F1 usa 2VP/(2VP+FP+FN), equivalente à média
        harmônica quando precisão e revocação permitem essa expressão.
        """
        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        return {
            "acuracia": (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn),
            "precisao": ratio(self.tp, self.tp + self.fp),
            "revocacao": ratio(self.tp, self.tp + self.fn),
            "especificidade": ratio(self.tn, self.tn + self.fp),
            "f1": ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
        }


def confusion_counts(targets: Sequence[int], predictions: Sequence[int]) -> ConfusionCounts:
    """Conta pares (y, classe) 0/1 alinhados; rejeita vazios ou formas diferentes.

    Não recebe probabilidades: aplique probability_to_class explicitamente antes.
    Assim a métrica não esconde a decisão que converteu uma estimativa em alerta.
    """
    validate_targets(targets, len(predictions))
    validate_targets(predictions, len(targets))
    pairs = list(zip(targets, predictions, strict=True))
    return ConfusionCounts(
        tn=pairs.count((0, 0)), fp=pairs.count((0, 1)),
        fn=pairs.count((1, 0)), tp=pairs.count((1, 1)),
    )


def majority_class(targets: Sequence[int]) -> int:
    """Escolhe baseline somente a partir de y de TREINO; empate favorece classe 1."""
    validate_targets(targets, len(targets))
    return int(sum(targets) >= len(targets) / 2)
