"""Interface local: analisar → revisar → salvar → filtrar → concluir.

Execute ``python -m sos_ml.rotina.app`` na raiz. A importação não inicia servidor.
Gradio fica neste módulo; o restante do aplicativo pode ser estudado/testado sem
biblioteca de interface e sem Ollama. A aplicação atende uma pessoa por processo.
"""

import argparse
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

from sos_ml.rotina.classifier import category_by_ml
from sos_ml.rotina.domain import CATEGORIES, PRIORITIES, Task, validate_tasks
from sos_ml.rotina.llm import MODELS
from sos_ml.rotina.rules import category_by_rules
from sos_ml.rotina.service import METHODS, suggest
from sos_ml.rotina.storage import COLUMNS, STATUSES, TaskStore

PREVIEW_HEADERS = ["Descrição", "Categoria", "Prioridade"]


def preview_rows(tasks: list[Task]) -> list[list[str]]:
    """Converte sugestões para três células editáveis, sem IDs ou gravação."""
    return [[task.descricao, task.categoria, task.prioridade] for task in tasks]


def tasks_from_rows(rows: object) -> list[Task]:
    """Revalida a tabela editada: a UI também é uma fonte de dados não confiável."""
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != 3 for row in rows
    ):
        raise ValueError("A revisão deve ter três colunas por tarefa.")
    return validate_tasks(
        [dict(zip(("descricao", "categoria", "prioridade"), row, strict=True)) for row in rows]
    )


def render_task_cards(rows: list[dict[str, str]]) -> str:
    """Monta cartões acessíveis com conteúdo escapado, incluindo estado vazio.

    Somente o estado validado determina classes CSS. Descrições podem conter
    símbolos HTML; escape impede que sejam interpretadas como marcação ou script.
    """
    if not rows:
        return (
            '<div class="empty-tasks"><span>○</span><h3>Espaço para seus próximos passos</h3>'
            "<p>Nenhuma tarefa por aqui. Envie uma mensagem na Conversa "
            "ou ajuste os filtros.</p></div>"
        )
    cards = []
    for row in rows:
        done = row["status"] == "concluída"
        marker = "✓" if done else "○"
        css_class = "task-card done" if done else "task-card"
        priority_class = "priority-high" if row["prioridade"] == "alta" else "priority-normal"
        cards.append(
            f'<article class="{css_class}"><span class="task-marker">{marker}</span><div>'
            f'<h3>{escape(row["descricao"])}</h3><div class="task-meta">'
            f"<span>{escape(row['categoria'])}</span>"
            f'<span class="{priority_class}">Prioridade {escape(row["prioridade"])}</span>'
            f"<span>{escape(row['status'])}</span></div></div></article>"
        )
    return '<div class="task-list">' + "".join(cards) + "</div>"


def build_app(store: TaskStore | None = None) -> Any:
    """Monta componentes e eventos; retorna Blocks sem abrir rede.

    ``store`` pode apontar para pasta temporária nos testes. Callbacks capturam
    erros esperados em mensagens em português. O estado da revisão é por sessão;
    o CSV é compartilhado pelas abas locais desta mesma pessoa.
    """
    import gradio as gr

    store = store if store is not None else TaskStore()

    def listing(category: str, status: str, priority: str) -> list[list[str]]:
        """Mantém a ordem das colunas consistente com o cabeçalho da lista."""
        return [
            [row[column] for column in COLUMNS]
            for row in store.list_tasks(category, status, priority)
        ]

    def analyze(text: str, method: str, model: str) -> tuple[Any, ...]:
        """Invalida revisão anterior também quando uma nova análise falha."""
        try:
            tasks, message = suggest(text, method, model)
            comparison = [
                [
                    task.descricao,
                    category_by_rules(task.descricao),
                    category_by_ml(task.descricao),
                    task.categoria,
                ]
                for task in tasks
            ]
            if not tasks:
                message += " Nenhuma tarefa identificada; descreva uma ação concreta."
            return preview_rows(tasks), comparison, message, bool(tasks)
        except (ValueError, OSError) as error:
            return [], [], f"Não foi possível analisar: {error}", False

    def save(
        rows: object, ready: bool, category: str, status: str, priority: str
    ) -> tuple[Any, ...]:
        """Exige análise prévia e valida revisão; limpa lote após confirmação."""
        try:
            if not ready:
                raise ValueError("Analise uma descrição antes de salvar.")
            tasks = tasks_from_rows(rows)
            store.add(tasks)
        except (ValueError, OSError) as error:
            # Preserva a revisão para correção, sem expor um traceback ao estudante.
            return gr.skip(), f"Não foi possível salvar: {error}", gr.skip(), ready
        return (
            listing(category, status, priority),
            ("Revisão salva. Descrições já pendentes não foram duplicadas."),
            [],
            False,
        )

    def refresh(category: str, status: str, priority: str) -> tuple[Any, str]:
        """Atualiza filtros sem persistir; arquivo inválido aparece como erro."""
        try:
            return listing(category, status, priority), "Lista atualizada."
        except (ValueError, OSError) as error:
            return [], f"Não foi possível ler o CSV: {error}"

    def complete(task_id: str, category: str, status: str, priority: str) -> tuple[Any, str]:
        """Conclui o ID informado e reaplica os filtros atuais."""
        try:
            store.complete(task_id.strip())
            return listing(category, status, priority), "Tarefa concluída."
        except (ValueError, OSError) as error:
            return gr.skip(), f"Não foi possível concluir: {error}"

    def workspace(category: str, status: str, priority: str) -> tuple[Any, ...]:
        """Atualiza cartões, contadores e seleção por nome mantendo o ID como valor.

        HTML recebe apenas textos escapados. Os contadores descrevem o arquivo
        inteiro; filtros mudam os cartões e as opções de conclusão, não os totais.
        """
        try:
            all_rows = store.list_tasks()
            rows = store.list_tasks(category, status, priority)
            pending = sum(row["status"] == "pendente" for row in all_rows)
            done = len(all_rows) - pending
            high = sum(
                row["status"] == "pendente" and row["prioridade"] == "alta" for row in all_rows
            )
            summary = (
                '<div class="routine-stats">'
                f"<span><b>{pending}</b> pendentes</span>"
                f"<span><b>{done}</b> concluídas</span>"
                f"<span><b>{high}</b> com prioridade alta</span></div>"
            )
            cards = render_task_cards(rows)
            choices = [
                (row["descricao"] + " · " + row["categoria"], row["id"])
                for row in rows
                if row["status"] == "pendente"
            ]
            return (
                summary,
                cards,
                gr.update(choices=choices, value=None),
                listing(category, status, priority),
                gr.update(interactive=bool(choices)),
            )
        except (ValueError, OSError):
            return (
                gr.skip(),
                '<p class="empty-tasks">Não foi possível abrir as tarefas. '
                "Confira a mensagem abaixo.</p>",
                gr.update(choices=[], value=None),
                [],
                gr.update(interactive=False),
            )

    def send_message(
        text: str, method: str, model: str, history: list[dict[str, str]]
    ) -> tuple[Any, ...]:
        """Acrescenta uma troca visual sem enviar histórico pessoal ao extrator.

        O serviço interpreta apenas a mensagem atual. As últimas 24 mensagens
        ficam na sessão do navegador; não são persistidas nem usadas como memória
        do LLM. Cada envio substitui a revisão anterior, inclusive quando falha.
        """
        rows, comparison_rows, notice, ready = analyze(text, method, model)
        if ready:
            count = len(rows)
            reply = f"Encontrei {count} tarefa{'s' if count != 1 else ''} para organizar.\n\n"
            reply += "\n".join(f"• {row[0]} — {row[1]} · prioridade {row[2]}" for row in rows)
            reply += "\n\nConfira as sugestões na revisão abaixo. Você pode editar antes de salvar."
        else:
            reply = "Não encontrei tarefas para revisar. Descreva uma ação, como estudar Python."
        reply += "\n\n" + notice
        # Não inclui entradas vazias e não limpa uma entrada recusada: fica fácil corrigi-la.
        conversation = list(history or [])
        if text.strip():
            conversation.append({"role": "user", "content": text})
        conversation.append({"role": "assistant", "content": reply})
        return (
            conversation[-24:],
            rows,
            comparison_rows,
            notice,
            ready,
            gr.update(visible=ready, open=ready),
            gr.update(interactive=ready),
            "" if ready else text,
        )

    def save_review(
        rows: object,
        ready: bool,
        category: str,
        status: str,
        priority: str,
        history: list[dict[str, str]],
    ) -> tuple[Any, ...]:
        """Confirma a revisão e dá retorno na conversa sem ocultar erros de edição."""
        table_rows, notice, cleared, still_ready = save(rows, ready, category, status, priority)
        conversation = list(history or [])
        if not still_ready and ready:
            conversation.append(
                {
                    "role": "assistant",
                    "content": (
                        "Tudo pronto: sua revisão foi salva. Abra Minhas tarefas "
                        "para acompanhar ou marcar uma tarefa como concluída. "
                        "Descrições já pendentes não são duplicadas."
                    ),
                }
            )
        return (
            conversation[-24:],
            cleared,
            notice,
            still_ready,
            gr.update(visible=still_ready, open=still_ready),
            gr.update(interactive=still_ready),
            *workspace(category, status, priority),
        )

    def refresh_workspace(category: str, status: str, priority: str) -> tuple[Any, ...]:
        """Reaplica filtros e atualiza contadores sem interferir na conversa."""
        _, notice = refresh(category, status, priority)
        return (*workspace(category, status, priority), notice)

    def complete_selected(
        task_id: str | None, category: str, status: str, priority: str
    ) -> tuple[Any, ...]:
        """A pessoa escolhe pelo nome; o armazenamento recebe o identificador estável."""
        if task_id is None:
            notice = "Escolha uma tarefa pendente para concluir."
        else:
            _, notice = complete(task_id, category, status, priority)
        return (*workspace(category, status, priority), notice)

    def new_conversation() -> tuple[Any, ...]:
        """Limpa só a sessão e abre Conversa, preservando o arquivo de tarefas."""
        return (
            [],
            [],
            [],
            "Nova conversa. Suas tarefas salvas continuam em Minhas tarefas.",
            False,
            gr.update(visible=False),
            gr.update(interactive=False),
            "",
            gr.update(selected="conversation"),
        )

    with gr.Blocks(
        title="Meu Assistente de Rotina",
        analytics_enabled=False,
        theme=gr.themes.Soft(
            primary_hue="emerald",
            neutral_hue="slate",
            font=["Arial", "sans-serif"],
            font_mono=["monospace"],
        ),
        js=files("sos_ml.rotina").joinpath("localizacao.js").read_text(encoding="utf-8"),
        css=files("sos_ml.rotina").joinpath("interface.css").read_text(encoding="utf-8"),
    ) as demo:
        with gr.Row(elem_id="routine-shell"):
            with gr.Column(scale=1, min_width=230, elem_id="routine-sidebar"):
                gr.HTML(
                    '<div class="brand"><span class="brand-icon">r.</span>'
                    "<div>Meu Assistente<small>DE ROTINA</small></div></div>"
                )
                new_button = gr.Button("＋ Nova conversa", elem_id="new-conversation")
                gr.Markdown("Seu dia, com um pouco mais de clareza.", elem_classes="sidebar-copy")
                with gr.Accordion("Como o assistente funciona", open=False):
                    method = gr.Radio(list(METHODS), value=METHODS[0], label="Como analisar")
                    model = gr.Dropdown(
                        list(MODELS), value=MODELS[0], label="Modelo do Ollama", visible=False
                    )
                    gr.Markdown(
                        "**Regras:** palavras e prioridades definidas em Python.\n\n"
                        "**ML:** aprende categorias com exemplos.\n\n"
                        "**LLM:** sugere tarefas usando o modelo local."
                    )
                gr.HTML(
                    '<div class="local-note"><span class="status-dot"></span>'
                    "Seu espaço local<p>Você revisa. O assistente organiza.</p></div>"
                )
            with gr.Column(scale=4, min_width=320, elem_id="routine-main"):
                gr.HTML(
                    '<div class="topbar"><span>UM PASSO DE CADA VEZ</span>'
                    '<span class="local-badge">Sem conta · no seu computador</span></div>'
                )
                summary = gr.HTML()
                with gr.Tabs() as tabs:
                    with gr.Tab("Conversa", id="conversation"):
                        chat = gr.Chatbot(
                            type="messages",
                            value=[],
                            show_label=False,
                            height=390,
                            layout="bubble",
                            render_markdown=False,
                            sanitize_html=True,
                            feedback_options=[],
                            show_share_button=False,
                            allow_file_downloads=False,
                            allow_tags=False,
                            elem_id="routine-chat",
                            placeholder='<div class="chat-welcome">'
                            '<span class="welcome-icon">✳</span>'
                            "<h1>Vamos organizar seu dia?</h1><p>Coloque suas ideias aqui. "
                            "Eu ajudo a transformar em tarefas.</p></div>",
                        )
                        with gr.Row(elem_id="suggestion-chips"):
                            examples = [
                                (
                                    "◷ Organizar os estudos",
                                    "Estudar Python; revisar exercício do curso",
                                ),
                                ("⌂ Cuidar da casa", "Lavar roupa; limpar cozinha sem pressa"),
                                (
                                    "♡ Ajudar a comunidade",
                                    "Separar doação; organizar campanha da ong",
                                ),
                            ]
                            example_buttons = [
                                (gr.Button(label, size="sm"), value) for label, value in examples
                            ]
                        with gr.Group(elem_id="routine-composer"):
                            text = gr.Textbox(
                                label="O que você precisa fazer?",
                                show_label=False,
                                lines=2,
                                max_lines=5,
                                max_length=2000,
                                container=False,
                                placeholder="Estudar Python; pagar conta hoje…",
                            )
                            with gr.Row():
                                gr.Markdown(
                                    "Separe as tarefas por **;** ou por linha.",
                                    elem_id="composer-hint",
                                )
                                send_button = gr.Button(
                                    "Enviar ↑", variant="primary", scale=0, min_width=110
                                )
                        gr.Markdown(
                            "Cada mensagem gera uma nova revisão. Nada é salvo sem você confirmar.",
                            elem_classes="microcopy",
                        )
                        ready = gr.State(False)
                        message = gr.Markdown("", elem_id="routine-feedback")
                        with gr.Accordion(
                            "Revise antes de guardar", open=True, visible=False
                        ) as review:
                            gr.Markdown(
                                "Dê dois cliques em uma célula para corrigir. "
                                "Remova uma linha pelo menu da célula."
                            )
                            preview = gr.Dataframe(
                                headers=PREVIEW_HEADERS,
                                datatype="str",
                                type="array",
                                col_count=(3, "fixed"),
                                interactive=True,
                                label="Suas sugestões",
                            )
                            gr.Markdown(
                                "**Categorias:** "
                                + ", ".join(CATEGORIES)
                                + ". **Prioridades:** "
                                + ", ".join(PRIORITIES)
                                + ".",
                                elem_classes="microcopy",
                            )
                            save_button = gr.Button(
                                "Confirmar e salvar tarefas", variant="primary", interactive=False
                            )
                        with gr.Accordion("Por dentro da IA · compare os métodos", open=False):
                            gr.Markdown(
                                "Compare as categorias da mesma descrição. "
                                "No modo ML, separar tarefas e definir prioridades "
                                "ainda são regras. "
                                "A comparação mostra a análise original, antes das suas edições."
                            )
                            comparison = gr.Dataframe(
                                headers=["Descrição", "Regras", "ML", "Método escolhido"],
                                datatype="str",
                                type="array",
                                interactive=False,
                                label="Comparação didática",
                            )
                            gr.Markdown(
                                "A conversa fica nesta sessão. Cada mensagem é interpretada "
                                "individualmente; o histórico não é memória do modelo."
                            )
                    with gr.Tab("Minhas tarefas", id="tasks"):
                        gr.Markdown(
                            "### Um lugar para o que importa\n"
                            "Acompanhe suas tarefas e celebre cada etapa concluída."
                        )
                        with gr.Row(elem_id="task-filters"):
                            category = gr.Dropdown(
                                ["Todas", *CATEGORIES], value="Todas", label="Categoria"
                            )
                            status = gr.Dropdown(
                                ["Todos", *STATUSES], value="Todos", label="Situação"
                            )
                            priority = gr.Dropdown(
                                ["Todas", *PRIORITIES], value="Todas", label="Prioridade"
                            )
                        cards = gr.HTML()
                        with gr.Group(elem_id="complete-task"):
                            task_id = gr.Dropdown(
                                choices=[],
                                label="Qual tarefa você terminou?",
                                value=None,
                                allow_custom_value=False,
                            )
                            complete_button = gr.Button(
                                "✓ Marcar como concluída", interactive=False
                            )
                        task_notice = gr.Markdown("", elem_id="task-feedback")
                        refresh_button = gr.Button("↻ Atualizar tarefas", size="sm")
                        with gr.Accordion("Ver os dados como tabela", open=False):
                            table = gr.Dataframe(
                                headers=["ID", "Descrição", "Categoria", "Prioridade", "Situação"],
                                datatype="str",
                                type="array",
                                interactive=False,
                                label="Dados do CSV",
                            )
                gr.Markdown(
                    "Use exemplos fictícios na aula. O assistente organiza tarefas; "
                    "você decide e executa as ações.",
                    elem_classes="microcopy",
                )
        filters = [category, status, priority]
        workspace_outputs = [summary, cards, task_id, table, complete_button]
        send_outputs = [chat, preview, comparison, message, ready, review, save_button, text]
        for event in [send_button.click, text.submit]:
            event(send_message, [text, method, model, chat], send_outputs, concurrency_id="rotina")
        new_button.click(new_conversation, outputs=[*send_outputs, tabs], concurrency_id="rotina")
        chat.clear(new_conversation, outputs=[*send_outputs, tabs], concurrency_id="rotina")
        for button, value in example_buttons:
            button.click(lambda value=value: value, outputs=text, queue=False)
        method.change(
            lambda value: gr.update(visible=value == METHODS[2]), method, model, queue=False
        )
        save_button.click(
            save_review,
            [preview, ready, *filters, chat],
            [chat, preview, message, ready, review, save_button, *workspace_outputs],
            concurrency_id="rotina",
        )
        refresh_button.click(
            refresh_workspace, filters, [*workspace_outputs, task_notice], concurrency_id="rotina"
        )
        for component in filters:
            component.change(
                refresh_workspace,
                filters,
                [*workspace_outputs, task_notice],
                concurrency_id="rotina",
            )
        complete_button.click(
            complete_selected,
            [task_id, *filters],
            [*workspace_outputs, task_notice],
            concurrency_id="rotina",
        )
        demo.load(
            refresh_workspace, filters, [*workspace_outputs, task_notice], concurrency_id="rotina"
        )
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    """Lê caminho/porta opcionais; inicia somente no computador local, sem link público."""
    parser = argparse.ArgumentParser(description="Meu Assistente de Rotina — aplicação local")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/rotina/tarefas.csv"),
        help="Caminho do CSV pessoal (padrão: data/rotina/tarefas.csv).",
    )
    parser.add_argument("--port", type=int, default=7860, help="Porta local (padrão: 7860).")
    args = parser.parse_args()
    try:
        demo = build_app(TaskStore(args.csv))
    except ImportError:
        parser.exit(1, 'Instale a interface: python -m pip install -e ".[rotina]"\n')
    print("Abra o navegador em http://127.0.0.1:" + str(args.port) + ". Para encerrar, Ctrl+C.")
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
