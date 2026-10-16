"""Escolha explícita do método, com fallback observável e sem efeitos em arquivos."""

from http.client import HTTPException

from sos_ml.rotina.classifier import category_by_ml
from sos_ml.rotina.domain import Task, validate_input
from sos_ml.rotina.llm import MODELS, extract_with_ollama
from sos_ml.rotina.rules import extract_by_rules

METHODS = ("Regras em Python", "Regras + ML", "LLM local (Ollama)")


def suggest(text: str, method: str, model: str = MODELS[0]) -> tuple[list[Task], str]:
    """Retorna tarefas e origem real; entrada inválida não aciona fallback.

    Em ML, separação e prioridade continuam sendo regras. Em LLM, falhas de rede
    ou de contrato voltam às regras puras, sem depender do classificador. Se as
    próprias regras não conseguirem representar a entrada, o erro segue para a UI.
    """
    validate_input(text)
    if method not in METHODS or model not in MODELS:
        raise ValueError("Escolha método e modelo disponíveis na interface.")
    if method == METHODS[2]:
        try:
            return extract_with_ollama(text, model), f"Sugestão do LLM local: {model}. Revise."
        except (OSError, HTTPException, ValueError, UnicodeError):
            return extract_by_rules(text), (
                "Ollama indisponível ou resposta inválida. Fallback: regras em Python. "
                "Confira ollama list, o servidor e a descrição; revise as tarefas."
            )
    tasks = extract_by_rules(text)
    if method == METHODS[1]:
        tasks = [
            Task(task.descricao, category_by_ml(task.descricao), task.prioridade) for task in tasks
        ]
    return tasks, f"Sugestão por {method}. Revise antes de salvar."
