"""CSV local de uma pessoa, sem banco ou sincronização entre processos.

Um lock protege cliques concorrentes no mesmo processo. Cada alteração grava um
arquivo temporário na mesma pasta e o substitui atomicamente; uma falha anterior
à substituição preserva o arquivo original. Não execute dois servidores no mesmo
CSV: o lock não coordena processos diferentes. Não edite o CSV com o app aberto.
"""

import csv
import os
import tempfile
from pathlib import Path
from threading import RLock

from sos_ml.rotina.domain import Task, validate_tasks

COLUMNS = ("id", "descricao", "categoria", "prioridade", "status")
STATUSES = ("pendente", "concluída")
_LOCK = RLock()


class TaskStore:
    """Repositório pequeno com IDs inteiros estáveis e status em português.

    O caminho padrão é relativo à raiz de execução e ignorado pelo Git. O arquivo
    só nasce ao salvar; diretórios pais são criados nesse momento. CSV corrompido
    é recusado, nunca tratado como lista vazia que poderia sobrescrever dados.
    """

    def __init__(self, path: Path = Path("data/rotina/tarefas.csv")) -> None:
        """Define destino local; não lê nem escreve durante a construção."""
        self.path = path

    def _read(self) -> list[dict[str, str]]:
        """Carrega e valida todo o contrato persistido, inclusive IDs únicos."""
        if not self.path.exists():
            return []
        with self.path.open(newline="", encoding="utf-8") as stream:
            try:
                # O modo permissivo aceita, por exemplo, texto depois de fechar
                # aspas e pode mudar a descrição ao regravar. Exigimos sintaxe
                # estrita antes de validar o significado de cada campo abaixo.
                reader = csv.DictReader(stream, strict=True)
                if reader.fieldnames != list(COLUMNS):
                    raise ValueError(
                        "Cabeçalho CSV inválido; preserve o arquivo e confira as colunas."
                    )
                rows = list(reader)
            except csv.Error as error:
                # A interface já trata ValueError. Não transforme um erro de
                # leitura em lista vazia: add/complete devem preservar o arquivo.
                raise ValueError(
                    "O arquivo contém uma estrutura CSV inválida; preserve-o para inspeção."
                ) from error
        ids = set()
        for row in rows:
            if set(row) != set(COLUMNS) or any(not isinstance(v, str) for v in row.values()):
                raise ValueError("Há uma linha CSV incompleta ou com colunas extras.")
            task_id = row["id"]
            if not task_id.isascii() or not task_id.isdecimal() or int(task_id) < 1:
                raise ValueError("O CSV contém um identificador inválido.")
            if str(int(task_id)) != task_id or task_id in ids:
                raise ValueError("Os identificadores CSV devem ser únicos e sem zeros iniciais.")
            ids.add(task_id)
            Task(row["descricao"], row["categoria"], row["prioridade"])
            if row["status"] not in STATUSES:
                raise ValueError("O CSV contém um status inválido.")
        return rows

    def _write(self, rows: list[dict[str, str]]) -> None:
        """Escreve UTF-8 com quoting CSV correto; limpa temporário mesmo se falhar."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".tarefas-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def list_tasks(
        self,
        category: str = "Todas",
        status: str = "Todos",
        priority: str = "Todas",
    ) -> list[dict[str, str]]:
        """Filtra por igualdade, sem alterar os IDs usados para concluir tarefas."""
        with _LOCK:
            return [
                row
                for row in self._read()
                if (category == "Todas" or row["categoria"] == category)
                and (status == "Todos" or row["status"] == status)
                and (priority == "Todas" or row["prioridade"] == priority)
            ]

    def add(self, tasks: list[Task]) -> list[dict[str, str]]:
        """Revalida lote inteiro e acrescenta tarefas pendentes, sem salvar parcialmente.

        Tarefas idênticas já pendentes não são duplicadas. Uma descrição concluída
        pode ser cadastrada de novo (rotina recorrente), recebendo outro ID.
        """
        validated = validate_tasks(
            [
                {
                    "descricao": task.descricao,
                    "categoria": task.categoria,
                    "prioridade": task.prioridade,
                }
                for task in tasks
            ]
        )
        if not validated:
            raise ValueError("Não há tarefas revisadas para salvar.")
        with _LOCK:
            rows = self._read()
            next_id = max((int(row["id"]) for row in rows), default=0) + 1
            for task in validated:
                if any(
                    row["descricao"].casefold() == task.descricao.casefold()
                    and row["status"] == "pendente"
                    for row in rows
                ):
                    continue
                rows.append(
                    {
                        "id": str(next_id),
                        "descricao": task.descricao,
                        "categoria": task.categoria,
                        "prioridade": task.prioridade,
                        "status": "pendente",
                    }
                )
                next_id += 1
            self._write(rows)
            return rows

    def complete(self, task_id: str) -> None:
        """Conclui somente o ID exato, mesmo quando a tabela está filtrada.

        ID ausente gera ValueError sem escrita; repetir conclusão é idempotente.
        Não há remoção de tarefas nesta versão, preservando um histórico simples.
        """
        with _LOCK:
            rows = self._read()
            for row in rows:
                if row["id"] == task_id:
                    row["status"] = "concluída"
                    self._write(rows)
                    return
            raise ValueError("ID não encontrado. Copie o ID da tarefa na lista.")
