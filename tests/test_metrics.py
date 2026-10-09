"""Contas pequenas verificam orientação da matriz, razões e casos indefinidos."""

import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sos_ml.from_scratch.metrics import ConfusionCounts, confusion_counts, majority_class


def test_counts_and_formulas() -> None:
    """Três VN, um FP, dois FN e quatro VP somam dez observações."""
    y = [0] * 4 + [1] * 6
    predicted = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    counts = confusion_counts(y, predicted)
    assert counts.matrix == [[3, 1], [2, 4]]
    metrics = counts.metrics()
    assert metrics == pytest.approx({
        "acuracia": 0.7, "precisao": 0.8, "revocacao": 2 / 3,
        "especificidade": 0.75, "f1": 8 / 11,
    })
    for name, reference in (
        ("acuracia", accuracy_score), ("precisao", precision_score),
        ("revocacao", recall_score), ("f1", f1_score),
    ):
        assert metrics[name] == pytest.approx(reference(y, predicted))


def test_imbalanced_and_undefined() -> None:
    """90% de acertos com nenhuma falha detectada não significa bom detector."""
    counts = confusion_counts([0] * 90 + [1] * 10, [0] * 100)
    assert counts.metrics()["acuracia"] == 0.9
    assert counts.metrics()["revocacao"] == 0
    assert counts.metrics()["precisao"] is None
    assert ConfusionCounts(2, 0, 0, 0).metrics()["f1"] is None
    assert majority_class([0, 0, 1]) == 0
    assert majority_class([0, 1]) == 1


@pytest.mark.parametrize("y, predictions", [([], []), ([1], []), ([2], [1])])
def test_invalid_labels(y: list[int], predictions: list[int]) -> None:
    """Dados inválidos não são convertidos silenciosamente em contagens."""
    with pytest.raises(ValueError):
        confusion_counts(y, predictions)
