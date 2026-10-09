"""Política, concentração, consequências e fallback não dependem de provedor remoto."""

import pytest

from sos_ml.decisions import (
    ACTIONS,
    EquipmentState,
    LocalDecisionModel,
    StructuredDecision,
    apply_policy,
    evaluate_decision,
)
from sos_ml.neural_training import train_equipment


@pytest.mark.parametrize(
    "probability, choice, human",
    [
        (0.1, "operacao_normal", False),
        (0.4, "agendar_inspecao", True),
        (0.68, "agendar_inspecao", False),
        (0.95, "revisao_humana_imediata", True),
    ],
)
def test_policy_examples(probability, choice, human) -> None:
    """Modelos podem sugerir, mas a aplicação decide quando pedir revisão."""
    decision = StructuredDecision(choice, probability, abs(2 * probability - 1))
    assert apply_policy(EquipmentState(70, 4), decision).requires_human is human
    assert apply_policy(EquipmentState(70, 4, True), decision).requires_human


def test_outside_domain_and_choices() -> None:
    """Abstenção não produz uma previsão inventada, e fallback não pode ser omitido."""
    provider = LocalDecisionModel(train_equipment(epochs=1).artifact)
    result = provider.decide(EquipmentState(150, 4))
    assert result.abstained and result.probability is None
    with pytest.raises(ValueError):
        provider.decide(EquipmentState(70, 4), ACTIONS[:2])


def test_provider_failure_requires_review() -> None:
    """Uma fronteira mockada verifica o fallback sem depender de disponibilidade externa."""

    class FailedProvider:
        def decide(self, state, choices):
            raise RuntimeError("Falha simulada")

    decision, policy = evaluate_decision(FailedProvider(), EquipmentState(70, 4))
    assert decision.abstained
    assert policy.requires_human


@pytest.mark.parametrize(
    "temperature, vibration", [(float("nan"), 1), (70, -1), (70, float("inf"))]
)
def test_invalid_state(temperature, vibration) -> None:
    """Dados inválidos são recusados antes do forward."""
    with pytest.raises(ValueError):
        EquipmentState(temperature, vibration)


def test_inconsistent_concentration_rejected() -> None:
    """A fronteira local não aceita p=0,5 descrito como concentração alta."""
    with pytest.raises(ValueError, match="Concentração"):
        StructuredDecision("operacao_normal", 0.5, 0.99)
    # Mesmo sugestão imprópria de operação não supera a regra de probabilidade alta.
    decision = StructuredDecision("operacao_normal", 0.95, 0.9)
    assert apply_policy(EquipmentState(70, 4), decision).requires_human
