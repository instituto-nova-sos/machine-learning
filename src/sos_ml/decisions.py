"""Decisão estruturada educacional local e política determinística de revisão.

Não é Jev nem modelo industrial. Um classificador estima p(falha sintética);
este adaptador organiza a estimativa num contrato com espaço de ações limitado.
A política pertence à aplicação e pode recusar a recomendação do modelo.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .equipment_io import EquipmentArtifact

Action = Literal["operacao_normal", "agendar_inspecao", "revisao_humana_imediata"]
ACTIONS: tuple[Action, ...] = (
    "operacao_normal",
    "agendar_inspecao",
    "revisao_humana_imediata",
)


@dataclass(frozen=True)
class EquipmentState:
    """Estado com temperatura em °C, vibração em mm/s e consequência da ação.

    Números devem ser finitos; vibração negativa é inválida. Faixa fora do
    gerador não é rejeitada como sintaxe: será sinalizada como fora do domínio.
    high_consequence=True exige revisão mesmo quando a estimativa é concentrada.
    Esse contexto é uma decisão de domínio, não atributo aprendido da rede.
    """

    temperature: float
    vibration: float
    high_consequence: bool = False

    def __post_init__(self) -> None:
        """Valida dados antes de qualquer inferência; ValueError indica erro de contrato."""
        if (
            not math.isfinite(self.temperature)
            or not math.isfinite(self.vibration)
            or self.vibration < 0
            or not isinstance(self.high_consequence, bool)
        ):
            raise ValueError(
                "Estado exige medidas finitas, vibração não negativa e contexto booleano."
            )

    @property
    def in_domain(self) -> bool:
        """Confere somente faixas do gerador, não detecta toda mudança de distribuição."""
        return 40 <= self.temperature <= 100 and 0.5 <= self.vibration <= 8


@dataclass(frozen=True)
class StructuredDecision:
    """Saída limitada: recomendação, p(falha), concentração e motivo de abstenção.

    confidence local é |2p−1|: zero para p=0,5 e um nos extremos. Não é uma
    probabilidade de que a ação esteja correta, nem reproduz o serviço Jev.
    Sem inferência (fora de domínio), probability=None, confidence=0 e abstained=True.
    """

    choice: Action
    probability: float | None
    confidence: float
    abstained: bool = False

    def __post_init__(self) -> None:
        """Valida domínio numérico e proíbe abstenção com aparência de ação automática."""
        if (
            self.choice not in ACTIONS
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Decisão exige ação conhecida e concentração entre 0 e 1.")
        if self.probability is not None and (
            not math.isfinite(self.probability) or not 0 <= self.probability <= 1
        ):
            raise ValueError("Probabilidade deve ser finita e pertencer a [0,1].")
        if not isinstance(self.abstained, bool):
            raise ValueError("Abstenção deve ser booleana.")
        if self.probability is not None and not math.isclose(
            self.confidence, abs(2 * self.probability - 1), abs_tol=1e-12
        ):
            raise ValueError("Concentração local deve corresponder a |2p−1|.")
        if self.probability is None and self.confidence != 0:
            raise ValueError("Sem inferência, concentração deve ser zero.")
        if self.probability is None and not self.abstained:
            raise ValueError("Sem probabilidade, é necessário abster-se.")
        if self.abstained and self.choice != "revisao_humana_imediata":
            raise ValueError("Abstenção exige revisão humana.")


class DecisionModel(Protocol):
    """Fronteira de provedor; testes podem substituir inferência por respostas fixas."""

    def decide(self, state: EquipmentState, choices: Sequence[Action]) -> StructuredDecision:
        """Avalia estado sem executar efeitos externos; escolha deve pertencer ao contrato."""
        ...


@dataclass
class LocalDecisionModel:
    """Adaptador de classificador para decisões limitadas, inteiramente local.

    Não gera texto nem estima distribuição sobre ações: recomenda por limiares
    didáticos fixos 0,3 e 0,7 da probabilidade binária. Essas constantes não são
    segurança operacional e não foram selecionadas observando o teste.
    """

    artifact: EquipmentArtifact

    def decide(
        self, state: EquipmentState, choices: Sequence[Action] = ACTIONS
    ) -> StructuredDecision:
        """Infere em domínio; fora dele abstém-se antes de chamar o modelo.

        Exige as três opções conhecidas, incluindo fallback. Aceitar silenciosamente
        uma lista incompleta impediria representar a decisão necessária. Não altera
        parâmetros/scalers nem controla máquinas; só retorna um valor estruturado.
        """
        if len(choices) != 3 or set(choices) != set(ACTIONS):
            raise ValueError("Forneça exatamente as três ações, incluindo revisão humana.")
        if not state.in_domain:
            return StructuredDecision("revisao_humana_imediata", None, 0, True)
        p = self.artifact.predict_proba([[state.temperature, state.vibration]])[0]
        choice: Action = (
            "operacao_normal"
            if p < 0.3
            else "agendar_inspecao"
            if p < 0.7
            else "revisao_humana_imediata"
        )
        return StructuredDecision(choice, p, abs(2 * p - 1))


@dataclass(frozen=True)
class PolicyResult:
    """Resultado autorizado pela aplicação, com justificativa e revisão explícita."""

    action: Action
    requires_human: bool
    reason: str


def apply_policy(
    state: EquipmentState, decision: StructuredDecision, *, minimum_confidence: float = 0.3
) -> PolicyResult:
    """Regras determinísticas: domínio, consequência e ambiguidade precedem sugestão.

    Cutoff 0,3 é ilustrativo: exige p fora de (0,35;0,65) para concentrar a saída
    binária. Confiança alta não garante correção. A política só autoriza exibição
    de status/encaminhamento didático, não operação física de equipamento.
    Consequência alta sempre exige humano. Retorna valor, sem side effects.
    """
    if not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
        raise ValueError("Limiar de concentração deve estar entre 0 e 1.")
    reason = ""
    if not state.in_domain or decision.abstained:
        reason = "Entrada fora do domínio sintético ou provedor em abstenção."
    elif state.high_consequence:
        reason = "Consequência elevada exige confirmação humana, mesmo com confiança alta."
    elif decision.confidence < minimum_confidence:
        reason = "Estimativa ambígua: solicitar revisão em vez de automatizar."
    elif decision.choice == "revisao_humana_imediata" or (
        decision.probability is not None and decision.probability >= 0.7
    ):
        reason = "Recomendação exige revisão; a aplicação não interrompe máquinas."
    if reason:
        return PolicyResult("revisao_humana_imediata", True, reason)
    return PolicyResult(decision.choice, False, "Exibição didática de baixo impacto autorizada.")


def evaluate_decision(
    model: DecisionModel, state: EquipmentState
) -> tuple[StructuredDecision, PolicyResult]:
    """Compõe provedor e política; falha de inferência leva a revisão explícita.

    ValueError/RuntimeError representam entradas/respostas inválidas ou falha do
    runtime. Não fazemos fallback silencioso para operação normal. Outros erros
    de programação não são ocultados; devem ser investigados pelos testes.
    """
    try:
        decision = model.decide(state, ACTIONS)
    except (ValueError, RuntimeError):
        decision = StructuredDecision("revisao_humana_imediata", None, 0, True)
    return decision, apply_policy(state, decision)
