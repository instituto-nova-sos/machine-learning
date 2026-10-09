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
