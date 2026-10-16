"""Contratos determinísticos do assistente; nenhuma chamada real ao Ollama."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sos_ml.rotina.app import preview_rows, tasks_from_rows
from sos_ml.rotina.classifier import category_by_ml, load_examples, train_classifier
from sos_ml.rotina.domain import Task, validate_tasks
from sos_ml.rotina.llm import MODELS, extract_with_ollama, parse_response
from sos_ml.rotina.rules import category_by_rules, extract_by_rules, priority_by_rules
from sos_ml.rotina.service import METHODS, suggest
from sos_ml.rotina.storage import TaskStore


def test_rules_split_accents_and_priorities() -> None:
    """Regras preservam coordenação dentro de tarefa e normalizam só a comparação."""
    tasks = extract_by_rules("Pagar boleto hoje; estudar Python\nComprar arroz e feijão sem pressa")
    assert [t.categoria for t in tasks] == ["finanças", "estudos", "outros"]
    assert [t.prioridade for t in tasks] == ["alta", "média", "baixa"]
    assert tasks[-1].descricao == "Comprar arroz e feijão sem pressa"
    assert category_by_rules("atualizar CURRÍCULO") == "emprego"
    assert category_by_rules("contato com amigo") == "outros"
    assert priority_by_rules("não é urgente") == "alta"  # Limitação conhecida, documentada.


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  ",
        "x" * 2001,
        "x" * 201,
        ";".join(["a"] * 11),
        "=fórmula",
        "pagar\x00boleto",
        "pagar; @comando",
    ],
)
def test_rules_reject_invalid_input(text: str) -> None:
    """Falhas não truncam entrada nem descartam silenciosamente tarefas excedentes."""
    with pytest.raises(ValueError):
        extract_by_rules(text)


@pytest.mark.parametrize(
    ("separator", "description_length", "extra_last_character"),
    [("\n", 199, False), ("\n", 199, True), ("\r\n", 198, False)],
)
def test_line_breaks_preserve_valid_input_budget(
    separator: str, description_length: int, extra_last_character: bool
) -> None:
    """Lotes válidos próximos de 2000 caracteres não crescem na normalização.

    LF cobre 1999 e exatamente 2000 caracteres; CRLF cobre 1998 caracteres.
    Cada descrição também respeita seu teto de 200. O resultado deve ser igual
    ao das mesmas tarefas separadas por ponto e vírgula, sem truncar conteúdo.
    """
    descriptions = [
        f"Estudar tarefa {index} ".ljust(description_length, "a") for index in range(10)
    ]
    if extra_last_character:
        descriptions[-1] += "a"
    text = separator.join(descriptions)
    assert 1998 <= len(text) <= 2000
    tasks = extract_by_rules(text)
    assert tasks == extract_by_rules(";".join(descriptions))
    assert [task.descricao for task in tasks] == descriptions


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [None],
        [{"descricao": "ok"}],
        [{"descricao": "ok", "categoria": "casa", "prioridade": "alta", "comando": "x"}],
        [{"descricao": "ok", "categoria": "saúde", "prioridade": "alta"}],
        [{"descricao": "ok", "categoria": "casa", "prioridade": 1}],
        [{"descricao": False, "categoria": "casa", "prioridade": "alta"}],
        [{"descricao": "=1+1", "categoria": "casa", "prioridade": "alta"}],
        [{"descricao": "ok", "categoria": "casa", "prioridade": "alta"}] * 11,
    ],
)
def test_validate_rejects_untrusted_data(value: object) -> None:
    """JSON válido sintaticamente ainda pode violar o contrato de domínio."""
    with pytest.raises(ValueError):
        validate_tasks(value)


@pytest.mark.parametrize(
    "content",
    ["```json\n{}\n```", "[]", '{"tarefas":[],"x":1}', '{"tarefas":"texto"}', "texto livre"],
)
def test_parse_rejects_invalid_response(content: str) -> None:
    """Resposta generativa não é corrigida ou executada como código."""
    with pytest.raises(ValueError):
        parse_response(content)


def test_preview_roundtrip_and_validation() -> None:
    """Edição humana também passa pelo contrato antes da persistência."""
    tasks = [Task("Lavar roupa", "casa", "baixa")]
    assert tasks_from_rows(preview_rows(tasks)) == tasks
    assert tasks_from_rows([["Estudar Python", "estudos", "alta"]])[0].prioridade == "alta"
    for rows in (None, [["incompleto"]], [["ok", "casa", "urgente"]]):
        with pytest.raises(ValueError):
            tasks_from_rows(rows)


def test_csv_roundtrip_filter_completion_and_duplicates(tmp_path: Path) -> None:
    """IDs são estáveis após filtros, reinício e cadastro de tarefa recorrente."""
    path = tmp_path / "pessoal" / "tarefas.csv"
    store = TaskStore(path)
    assert store.list_tasks() == [] and not path.exists()
    tasks = [
        Task('Lavar roupa, separar "toalhas"', "casa", "alta"),
        Task("Estudar Python", "estudos", "média"),
    ]
    store.add(tasks)
    store.add(tasks)
    assert len(store.list_tasks()) == 2
    assert store.list_tasks("estudos", "pendente", "média")[0]["id"] == "2"
    TaskStore(path).complete("2")
    assert store.list_tasks(status="concluída")[0]["descricao"] == "Estudar Python"
    store.complete("2")
    store.add([tasks[1]])
    assert [row["id"] for row in store.list_tasks()] == ["1", "2", "3"]
    assert TaskStore(path).list_tasks()[0]["descricao"] == tasks[0].descricao
    before = path.read_bytes()
    with pytest.raises(ValueError):
        store.complete("999")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        "",
        "coluna\nvalor\n",
        "id,descricao,categoria,prioridade,status\n1,Estudar,estudos,média,pendente\n"
        "1,Lavar,casa,baixa,pendente\n",
        "id,descricao,categoria,prioridade,status\n1,Estudar,estudos,média\n",
        "id,descricao,categoria,prioridade,status\n1,=1+1,casa,alta,pendente\n",
    ],
)
def test_corrupt_csv_is_preserved(tmp_path: Path, content: str) -> None:
    """Arquivo corrompido não pode ser sobrescrito por uma gravação aparentemente válida."""
    path = tmp_path / "tarefas.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        TaskStore(path).add([Task("Lavar roupa", "casa", "alta")])
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize(
    "line", ['1,"Lavar roupa"x,casa,baixa,pendente', '1,"Lavar roupa,casa,baixa,pendente']
)
@pytest.mark.parametrize("operation", ["listar", "adicionar", "concluir"])
def test_malformed_csv_quotes_are_rejected_without_rewriting(
    tmp_path: Path, line: str, operation: str
) -> None:
    """Aspas malformadas não podem ser corrigidas silenciosamente pelo leitor.

    Confere os bytes, incluindo CRLF, após leitura e tentativas de alteração.
    ValueError mantém o contrato já tratado pela interface, sem deixar csv.Error
    escapar ou trocar a descrição original por uma interpretação permissiva.
    """
    path = tmp_path / "tarefas.csv"
    original = ("id,descricao,categoria,prioridade,status\r\n" + line + "\r\n").encode("utf-8")
    path.write_bytes(original)
    store = TaskStore(path)
    with pytest.raises(ValueError, match="estrutura CSV inválida"):
        if operation == "listar":
            store.list_tasks()
        elif operation == "adicionar":
            store.add([Task("Estudar Python", "estudos", "média")])
        else:
            store.complete("1")
    assert path.read_bytes() == original


def test_atomic_write_failure_keeps_original(tmp_path: Path) -> None:
    """Falha na substituição mantém bytes anteriores e remove temporário."""
    path = tmp_path / "tarefas.csv"
    store = TaskStore(path)
    store.add([Task("Estudar", "estudos", "alta")])
    before = path.read_bytes()
    with patch.object(Path, "replace", side_effect=OSError("falha simulada")):
        with pytest.raises(OSError):
            store.add([Task("Lavar", "casa", "baixa")])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_adds_keep_unique_ids(tmp_path: Path) -> None:
    """Duas instâncias no mesmo processo compartilham o lock de leitura/escrita."""
    path = tmp_path / "tarefas.csv"
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda i: TaskStore(path).add([Task(f"Estudar capítulo {i}", "estudos", "média")]),
                range(8),
            )
        )
    rows = TaskStore(path).list_tasks()
    assert len(rows) == len({row["id"] for row in rows}) == 8


def test_classifier_keeps_test_outside_vocabulary() -> None:
    """A transformação usa somente treino; dados desconhecidos recebem outros."""
    train_texts, labels = load_examples("treino")
    test_texts, _ = load_examples("teste")
    assert len(train_texts) == 42 and len(set(labels)) == 7 and len(test_texts) == 14
    assert not set(train_texts).intersection(test_texts)
    vocabulary = train_classifier().named_steps["palavras"].vocabulary_
    assert "transporte" not in vocabulary  # Palavra presente somente no teste.
    assert category_by_ml("pagar boleto") == "finanças"
    assert category_by_ml("zzzzxy") == "outros"
    assert train_classifier() is train_classifier()


@pytest.mark.parametrize(
    "failure", [OSError("sem servidor"), TimeoutError(), ValueError("JSON inválido")]
)
def test_fallback_is_explicit_and_deterministic(failure: Exception) -> None:
    """Falha no LLM retorna mesmas regras, sem salvar ou depender do ML."""
    with patch("sos_ml.rotina.service.extract_with_ollama", side_effect=failure):
        tasks, message = suggest("pagar boleto hoje", METHODS[2])
    assert tasks == extract_by_rules("pagar boleto hoje")
    assert "Fallback" in message


def test_modes_and_input_before_network() -> None:
    """Modo ML altera só categoria; entradas inválidas nem chegam ao Ollama."""
    assert suggest("pagar boleto", METHODS[0])[0][0].categoria == "finanças"
    assert suggest("pagar boleto", METHODS[1])[0][0].categoria == "finanças"
    with patch("sos_ml.rotina.service.extract_with_ollama") as extract:
        with pytest.raises(ValueError):
            suggest("", METHODS[2])
        extract.assert_not_called()
        extract.return_value = []
        assert suggest("não há tarefas", METHODS[2])[0] == []
    with pytest.raises(ValueError):
        suggest("Estudar", "método inexistente")


def mock_connection(envelope: object) -> MagicMock:
    """Constrói resposta HTTP em memória para testar o protocolo sem rede."""
    connection = MagicMock()
    response = connection.getresponse.return_value
    response.status = 200
    response.read.return_value = json.dumps(envelope).encode("utf-8")
    return connection


def test_ollama_request_and_response_contract() -> None:
    """Confere modelo, esquema, ausência de streaming/pensamento e fechamento HTTP."""
    tasks = [{"descricao": "Estudar Python", "categoria": "estudos", "prioridade": "média"}]
    connection = mock_connection(
        {"done": True, "message": {"content": json.dumps({"tarefas": tasks})}}
    )
    with patch("sos_ml.rotina.llm.HTTPConnection", return_value=connection) as factory:
        assert extract_with_ollama("Preciso estudar Python", MODELS[1]) == validate_tasks(tasks)
    factory.assert_called_once_with("127.0.0.1", 11434, timeout=60)
    args = connection.request.call_args.args
    assert args[:2] == ("POST", "/api/chat")
    payload = json.loads(args[2])
    assert payload["model"] == MODELS[1] and payload["format"]["type"] == "object"
    assert payload["stream"] is False and payload["think"] is False
    assert payload["messages"][-1]["content"] == "Preciso estudar Python"
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    "envelope", [{}, [], {"done": False}, {"done": True, "message": {"content": 1}}]
)
def test_ollama_bad_envelope(envelope: object) -> None:
    """Envelope incompleto ou malformado falha antes de acessar campos internos."""
    connection = mock_connection(envelope)
    with patch("sos_ml.rotina.llm.HTTPConnection", return_value=connection):
        with pytest.raises(ValueError):
            extract_with_ollama("Estudar")
    connection.close.assert_called_once()


def test_ollama_http_and_size_limits() -> None:
    """Erro HTTP e resposta excessiva são recusados e conexão é liberada."""
    for status, body in [(404, b""), (200, b"x" * 65537), (200, b"nao-json")]:
        connection = mock_connection({})
        connection.getresponse.return_value.status = status
        connection.getresponse.return_value.read.return_value = body
        with patch("sos_ml.rotina.llm.HTTPConnection", return_value=connection):
            with pytest.raises(ValueError):
                extract_with_ollama("Estudar")
        connection.close.assert_called_once()
