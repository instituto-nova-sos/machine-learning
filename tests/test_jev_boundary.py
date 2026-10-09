"""Valida fronteira do provedor com fixture local; jamais consulta disponibilidade Jev."""

import pytest

from sos_ml.jev_boundary import optional_policy, parse_choice


def fixture() -> dict[str, object]:
    """Resposta inventada de Choice, sem atribuir medições ao serviço real."""
    return {
        "type": "choice",
        "choice": "agendar_inspecao",
        "confidence": 0.7,
        "probabilities": {
            "operacao_normal": 0.1,
            "agendar_inspecao": 0.8,
            "revisao_humana_imediata": 0.1,
        },
    }


def test_valid_choice_and_review() -> None:
    """Consequência alta sempre exige revisão, e concentração baixa também."""
    answer = parse_choice(fixture())
    assert optional_policy(answer).requires_human
    assert optional_policy(answer, high_consequence=False).requires_human


@pytest.mark.parametrize(
    "field, value",
    [
        ("choice", "acao_desconhecida"),
        ("confidence", float("nan")),
        ("probabilities", {"operacao_normal": 1}),
        ("type", "noul"),
        (
            "probabilities",
            {"operacao_normal": 0.2, "agendar_inspecao": 0.2, "revisao_humana_imediata": 0.2},
        ),
    ],
)
def test_invalid_provider_answer(field, value) -> None:
    """Formato inválido é recusado antes de efeitos externos."""
    response = fixture()
    response[field] = value
    with pytest.raises(ValueError):
        parse_choice(response)


def test_forged_confidence_cannot_bypass_review() -> None:
    """Distribuição difusa não pode autorizar exibição alegando confiança máxima."""
    response = fixture()
    response["choice"] = "operacao_normal"
    response["probabilities"] = {
        "operacao_normal": 0.34,
        "agendar_inspecao": 0.33,
        "revisao_humana_imediata": 0.33,
    }
    response["confidence"] = 1.0
    with pytest.raises(ValueError, match="concentração"):
        optional_policy(parse_choice(response), high_consequence=False)


@pytest.mark.parametrize("maximum, confidence", [(1 / 3, 0.0), (0.8, 0.7), (1.0, 1.0)])
def test_confidence_matches_distribution(maximum: float, confidence: float) -> None:
    """Uniforme, exemplo da aula e distribuição pontual verificam a fórmula Choice."""
    response = fixture()
    remainder = maximum if confidence == 0 else (1 - maximum) / 2
    response["probabilities"] = {
        "operacao_normal": remainder,
        "agendar_inspecao": maximum,
        "revisao_humana_imediata": remainder,
    }
    response["confidence"] = confidence
    answer = parse_choice(response)
    assert answer.confidence == pytest.approx(confidence)
    assert optional_policy(answer, high_consequence=False).requires_human == (confidence < 0.8)
