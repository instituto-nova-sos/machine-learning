"""Regras explícitas em Python puro: nenhuma palavra ou prioridade é aprendida."""

import re
import unicodedata

from sos_ml.rotina.domain import MAX_TASKS, Task, clean_text, validate_input

# A ordem resolve empates. Edite aqui para estudar como uma política altera saídas.
KEYWORDS = {
    "finanças": ("pagar", "conta", "boleto", "orçamento", "dinheiro", "banco"),
    "estudos": ("estudar", "prova", "curso", "revisar", "aula", "exercício"),
    "emprego": ("currículo", "vaga", "entrevista", "emprego", "candidatura"),
    "negócio": ("cliente", "estoque", "venda", "loja", "fornecedor"),
    "casa": ("limpar", "lavar", "cozinha", "mercado", "casa", "roupa"),
    "voluntariado": ("doação", "doações", "voluntário", "comunidade", "ong"),
}


def normalize(text: str) -> str:
    """Remove acentos e caixa para comparar palavras sem alterar o texto salvo."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )


def category_by_rules(text: str) -> str:
    """Primeira categoria com palavra inteira conhecida; sem ocorrência usa outros.

    Usar palavras inteiras evita confundir 'conta' com 'contato'. Ainda não há
    interpretação de negação, contexto, sinônimos gerais ou múltiplas categorias.
    """
    words = set(re.findall(r"\w+", normalize(text)))
    for category, keywords in KEYWORDS.items():
        if words.intersection(normalize(word) for word in keywords):
            return category
    return "outros"


def priority_by_rules(text: str) -> str:
    """Hoje/urgente dá alta; 'sem pressa' dá baixa; demais casos recebem média.

    É uma convenção didática, não um calendário: 'não é urgente' também aciona
    alta. A pessoa deve corrigir esse erro na revisão e discuti-lo em aula.
    """
    normalized = normalize(text)
    if re.search(r"\b(hoje|urgente)\b", normalized):
        return "alta"
    return "baixa" if "sem pressa" in normalized else "média"


def extract_by_rules(text: str) -> list[Task]:
    """Separa por ponto e vírgula/linha nova, preservando 'e' dentro da descrição.

    Não divide 'arroz e feijão'. Até dez partes são aceitas; excesso ou descrição
    acima de 200 caracteres produz erro, sem truncamento silencioso.
    """
    parts = [part.strip() for part in validate_input(text).split(";") if part.strip()]
    if not parts or len(parts) > MAX_TASKS:
        raise ValueError("Informe de uma a dez tarefas separadas por ponto e vírgula.")
    return [
        Task(clean_text(part), category_by_rules(part), priority_by_rules(part)) for part in parts
    ]
