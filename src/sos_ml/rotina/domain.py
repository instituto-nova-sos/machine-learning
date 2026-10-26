"""Contrato compartilhado por regras, ML, LLM, tabela editável e persistência.

Validar a estrutura impede dados malformados; não comprova que uma sugestão é
verdadeira, útil ou fiel à intenção. A revisão da pessoa continua necessária.
"""

from dataclasses import dataclass

CATEGORIES = (
    "finanças",
    "estudos",
    "emprego",
    "negócio",
    "casa",
    "voluntariado",
    "hobby",
    "saúde",
    "família",
    "espiritual",
    "projeto",
    "outros",
)
PRIORITIES = ("alta", "média", "baixa")
MAX_TASKS = 10
MAX_INPUT = 2000


def clean_text(value: object, limit: int = 200) -> str:
    """Exige texto não vazio e limitado; retorna sem espaços nas extremidades.

    Recusa controles e prefixos de fórmula para que o CSV possa ser aberto em
    planilhas sem executar fórmulas. Não converte números/bool em texto. Entradas
    inválidas levantam ValueError antes de qualquer escrita ou chamada ao modelo.
    """
    if not isinstance(value, str):
        raise ValueError("Use texto, sem números isolados ou valores nulos.")
    text = value.strip()
    if not text or len(text) > limit:
        raise ValueError(f"Use entre 1 e {limit} caracteres.")
    if any(ord(char) < 32 for char in text) or text[0] in "=+-@":
        raise ValueError("Remova controles e prefixos de fórmula (=, +, -, @).")
    return text


@dataclass(frozen=True)
class Task:
    """Sugestão imutável; sem ID/status até ser confirmada e salva.

    ``descricao`` tem 1–200 caracteres, ``categoria`` e ``prioridade`` pertencem
    às listas fechadas acima. Os mesmos contratos valem para todos os extratores.
    """

    descricao: str
    categoria: str
    prioridade: str

    def __post_init__(self) -> None:
        """Rejeita valores fora do contrato, inclusive construção direta inválida."""
        if clean_text(self.descricao) != self.descricao:
            raise ValueError("Remova espaços das extremidades da descrição.")
        if self.categoria not in CATEGORIES or self.prioridade not in PRIORITIES:
            raise ValueError("Escolha uma categoria e uma prioridade da lista permitida.")


def validate_tasks(value: object) -> list[Task]:
    """Valida lista JSON completa antes de processar qualquer item.

    Aceita de zero a dez objetos com exatamente três campos textuais; nenhum
    campo adicional (como comando ou caminho) é interpretado. Não salva parte de
    um lote inválido. Lista vazia significa que nenhuma tarefa foi identificada.
    """
    if not isinstance(value, list) or len(value) > MAX_TASKS:
        raise ValueError("A resposta deve conter uma lista de até dez tarefas.")
    tasks = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"descricao", "categoria", "prioridade"}:
            raise ValueError("Cada tarefa precisa de descrição, categoria e prioridade.")
        description = clean_text(item["descricao"])
        category, priority = item["categoria"], item["prioridade"]
        if not isinstance(category, str) or not isinstance(priority, str):
            raise ValueError("Categoria e prioridade devem ser textos.")
        tasks.append(Task(description, category, priority))
    return tasks


def validate_input(text: str) -> str:
    """Aceita até 2000 caracteres; linhas novas separam tarefas no modo de regras."""
    if not isinstance(text, str) or len(text) > MAX_INPUT:
        raise ValueError("Escreva até 2000 caracteres.")
    # Quebras de linha são úteis na entrada, mas não ficam dentro de células CSV.
    # Um separador vira outro sem acrescentar espaços: uma entrada que já cabe
    # no orçamento de caracteres não deve ultrapassá-lo só pela normalização.
    return clean_text(text.replace("\r\n", "\n").replace("\n", ";"), MAX_INPUT)
