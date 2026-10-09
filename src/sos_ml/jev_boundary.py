"""Validação offline da fronteira opcional Jev, sem SDK, conta ou chamada remota.

Probabilidades aqui são SOBRE OPÇÕES de Choice, não P(falha) da rede local.
Nunca transforme confidence ou score em probabilidade de falha por conveniência.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .decisions import ACTIONS, Action, PolicyResult


@dataclass(frozen=True)
class JevChoice:
    """Choice validada: opção conhecida, distribuição e concentração do provedor."""

    choice: Action
    probabilities: dict[str, float]
    confidence: float


def parse_choice(payload: Mapping[str, object]) -> JevChoice:
    """Valida Choice bruto antes de uma política, ou levanta ValueError.

    Exige tipo choice, todas as três ações, soma de probabilidades próxima de 1
    (tolerância 1e-6), valores [0,1] finitos e escolha entre máximos. Não autentica
    origem nem comprova calibração: a fronteira valida contrato, não verdade.
    """
    choice = payload.get("choice")
    distribution = payload.get("probabilities")
    confidence = payload.get("confidence")
    if (
        payload.get("type") != "choice"
        or choice not in ACTIONS
        or not isinstance(distribution, dict)
        or set(distribution) != set(ACTIONS)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
    ):
        raise ValueError("Resposta Choice incompatível com o contrato de ações.")
    values = {}
    for key, value in distribution.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Distribuição deve ter probabilidades finitas entre 0 e 1.")
        values[key] = float(value)
    if (
        not math.isclose(sum(values.values()), 1, abs_tol=1e-6)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or values[choice] < max(values.values())
    ):
        raise ValueError("Soma, escolha ou concentração incompatíveis.")
    return JevChoice(choice, values, float(confidence))


def optional_policy(answer: JevChoice, *, high_consequence: bool = True) -> PolicyResult:
    """Exibe uma sugestão ou pede humano; nunca executa uma ação física.

    Corte 0,8 é ilustrativo para esta interface externa e NÃO transferido da
    concentração binária local. Provedor diferente exige avaliação própria.
    Consequência elevada sempre pede revisão, qualquer que seja a confiança.
    """
    if high_consequence or answer.confidence < 0.8 or answer.choice == "revisao_humana_imediata":
        return PolicyResult(
            "revisao_humana_imediata",
            True,
            "Integração opcional: revisão requerida pela política da aplicação.",
        )
    return PolicyResult(answer.choice, False, "Somente exibição didática de baixo impacto.")
