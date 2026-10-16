() => {
    // O Gradio 5 mantém alguns rótulos internos em inglês mesmo com componentes
    // em português. Este adaptador só traduz controles, nunca dados das tarefas.
    const traducoes = {
        "Table actions": "Ações da tabela",
        "Add row": "Adicionar linha",
        "Add row above": "Adicionar linha acima",
        "Add row below": "Adicionar linha abaixo",
        "Delete row": "Remover linha",
        "Add column left": "Adicionar coluna à esquerda",
        "Add column right": "Adicionar coluna à direita",
        "Delete column": "Remover coluna",
        "Sort ascending": "Ordenar em ordem crescente",
        "Sort descending": "Ordenar em ordem decrescente",
        "Sort column": "Ordenar coluna",
        "Clear sort": "Limpar ordenação",
        "Filter": "Filtrar",
        "Clear filter": "Limpar filtro",
        "Copy": "Copiar",
        "Paste": "Colar",
        "Cut": "Recortar",
        "Select all": "Selecionar tudo",
        "Select cell": "Selecionar célula",
        "Select column": "Selecionar coluna",
        "Select row": "Selecionar linha",
        "Edit cell": "Editar célula",
        "Open cell menu": "Abrir menu da célula",
        "Clear": "Limpar",
        "Cancel": "Cancelar",
        "Confirm": "Confirmar",
        "Save": "Salvar",
        "Edit": "Editar",
        "Search": "Buscar",
        "Export": "Exportar",
        "Import": "Importar",
        "Loading...": "Carregando...",
        "chatbot conversation": "Conversa com o assistente",
        "Drop CSV or TSV files here to import data into dataframe":
            "Solte arquivos CSV ou TSV aqui para importar para a tabela"
    };
    const traduzir = () => {
        document.documentElement.lang = "pt-BR";
        document.querySelectorAll("#routine-feedback, #task-feedback").forEach(elemento => {
            elemento.setAttribute("role", "status");
            elemento.setAttribute("aria-live", "polite");
        });
        document.querySelectorAll("[aria-label], [title], button, [role='menuitem']")
            .forEach(elemento => {
                for (const atributo of ["aria-label", "title"]) {
                    const atual = elemento.getAttribute(atributo);
                    if (traducoes[atual]) elemento.setAttribute(atributo, traducoes[atual]);
                    if (atual?.startsWith("user's message:"))
                        elemento.setAttribute(atributo, atual.replace("user's message:", "Sua mensagem:"));
                    if (atual?.startsWith("bot's message:"))
                        elemento.setAttribute(atributo, atual.replace("bot's message:", "Resposta do assistente:"));
                }
                // Somente nós diretos de controles: não altera células ou entradas.
                for (const no of elemento.childNodes) {
                    if (no.nodeType !== Node.TEXT_NODE) continue;
                    const atual = no.textContent.trim();
                    if (traducoes[atual]) no.textContent = traducoes[atual];
                }
            });
    };
    traduzir();
    // O menu de linhas é montado depois do clique; observe novos controles.
    new MutationObserver(traduzir).observe(document.body, {
        childList: true, subtree: true
    });
}
