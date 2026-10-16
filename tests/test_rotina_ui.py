"""Integração opcional da conversa e das tarefas, sem iniciar servidor."""

from pathlib import Path

import pytest

from sos_ml.rotina.app import build_app, render_task_cards
from sos_ml.rotina.service import METHODS
from sos_ml.rotina.storage import TaskStore


def test_gradio_callbacks_review_save_filter_complete(tmp_path: Path) -> None:
    """Conversa exige revisão; seleção pelo nome conserva o ID no armazenamento."""
    pytest.importorskip("gradio")
    store = TaskStore(tmp_path / "tarefas.csv")
    demo = build_app(store)
    callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values() if fn.fn is not None}
    sent = callbacks["send_message"](
        "Estudar Python; pagar boleto hoje", METHODS[0], "qwen3:1.7b", []
    )
    chat, preview, comparison, message, ready, review, button, composer = sent
    assert ready and len(preview) == len(comparison) == 2 and "Regras" in message
    assert review["visible"] and button["interactive"] and composer == ""
    assert [entry["role"] for entry in chat] == ["user", "assistant"]
    assert "Encontrei 2 tarefas" in chat[-1]["content"]
    assert not store.path.exists()
    preview[0][2] = "baixa"
    saved = callbacks["save_review"](preview, True, "Todas", "Todos", "Todas", chat)
    history, cleared, message, ready, review, button, summary, cards, choices, table, done = saved
    assert len(table) == 2 and cleared == [] and ready is False
    assert not review["visible"] and not button["interactive"]
    assert "2</b> pendentes" in summary and "Estudar Python" in cards
    assert "Minhas tarefas" in history[-1]["content"]
    assert store.list_tasks()[0]["prioridade"] == "baixa" and done["interactive"]
    callbacks["save_review"](preview, False, "Todas", "Todos", "Todas", history)
    assert len(store.list_tasks()) == 2
    filtered = callbacks["refresh_workspace"]("finanças", "pendente", "alta")
    assert filtered[2]["choices"] == [("pagar boleto hoje · finanças", "2")]
    assert filtered[3][0][0] == "2"
    completed = callbacks["complete_selected"]("2", "Todas", "concluída", "Todas")
    assert completed[3][0][0] == "2" and completed[3][0][-1] == "concluída"
    assert completed[2]["choices"] == [] and not completed[4]["interactive"]
    assert "1</b> concluídas" in completed[0]
    reset = callbacks["new_conversation"]()
    assert reset[0] == [] and reset[4] is False
    assert reset[-1]["selected"] == "conversation"
    assert len(store.list_tasks()) == 2  # Nova conversa não apaga tarefas confirmadas.
    invalid = callbacks["send_message"]("", METHODS[0], "qwen3:1.7b", chat)
    assert invalid[4] is False and not invalid[5]["visible"]
    failed = callbacks["save_review"](
        [["Inválida", "categoria inexistente", "alta"]], True, "Todas", "Todos", "Todas", chat
    )
    assert "Não foi possível salvar" in failed[2] and failed[3] is True
    assert failed[0] == chat  # Nunca confirma na conversa algo que não salvou.
    demo.close()


def test_cards_escape_user_content_and_empty_state() -> None:
    """Conteúdo vindo da tarefa é texto; marcação não se transforma em código HTML."""
    assert "Nenhuma tarefa" in render_task_cards([])
    cards = render_task_cards(
        [
            {
                "id": "1",
                "descricao": '<img src=x onerror="alert(1)">',
                "categoria": "casa",
                "prioridade": "alta",
                "status": "pendente",
            }
        ]
    )
    assert "<img src=" not in cards and "&lt;img" in cards
    assert "priority-high" in cards
