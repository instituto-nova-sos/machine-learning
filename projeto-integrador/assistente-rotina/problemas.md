# Solução de problemas

[← Guia e instalação](README.md)

| Sintoma | O que conferir e fazer |
|---|---|
| `No module named sos_ml` | entre na raiz, ative a `.venv` e rode `python -m pip install -e ".[dev,rotina]"` |
| Interface pede Gradio | instale o extra `rotina` na mesma `.venv` usada para iniciar o app |
| `python` não encontrado | no Windows siga o [diagnóstico de executável e versão](../../docs/ambiente-local.md#windows-powershell); `python3.exe` depende da instalação; em Linux/macOS confira `python3` |
| PowerShell bloqueia ativação | use diretamente `.\.venv\Scripts\python.exe -m sos_ml.rotina.app`, sem alterar políticas da máquina |
| Instalação sem internet falha | faça a preparação antes da aula; use máquina preparada em dupla e fallback de regras |
| Porta 7860 ocupada | encerre sua instância anterior com Ctrl+C ou use `python -m sos_ml.rotina.app --port 7861` |
| Navegador não abre sozinho | copie `http://127.0.0.1:7860` para a barra de endereço; mantenha terminal aberto |
| Ollama não encontrado | instale o programa oficial, reabra terminal; o modo de regras continua disponível |
| Fallback aparece no modo LLM | confira `ollama list`, nome exato selecionado e servidor em `127.0.0.1:11434` |
| Ollama sem modelo | execute `ollama pull qwen3:1.7b`, ou `ollama pull qwen3:0.6b` e selecione a alternativa |
| `ollama serve` diz porta ocupada | servidor pode já estar ativo pelo app; não inicie outro na mesma porta |
| Computador lento ou sem memória | use regras/ML, feche apps pesados; 0.6B é alternativa experimental, não garantia |
| Só uma tarefa foi reconhecida nas regras | separe ações por ponto e vírgula ou linha nova; `e` não é separador automático |
| Dez tarefas ou tamanho excedido | divida em lotes menores; nenhuma descrição pode exceder 200 caracteres |
| Sugestão errada | corrija descrição, categoria ou prioridade na tabela antes de salvar |
| Categoria/prioridade recusada | use exatamente uma opção mostrada; acentos fazem parte dos valores do contrato |
| Descrição rejeitada como fórmula | remova prefixos `=`, `+`, `-`, `@`; escreva a ação em palavras, sem marcadores |
| Salvar não adicionou outra linha | uma descrição idêntica já pendente não é duplicada; confira filtros e status |
| Não consigo escolher tarefa para concluir | abra Minhas tarefas; escolha pendente ou Todos no filtro de situação; selecione pelo nome |
| Tarefa desapareceu após concluir | filtro de pendentes a ocultou; selecione Todos ou concluída |
| Conversa sumiu após recarregar | histórico fica só na sessão; as tarefas confirmadas continuam em Minhas tarefas |
| Não vejo seletor de modelo | abra Como o assistente funciona e selecione LLM local (Ollama) |
| Mudança em regra/exemplos não aparece | pare e reinicie o app; classificador é treinado uma vez por processo |
| CSV inválido ou com permissão negada | encerre app, faça cópia privada e confira cabeçalho/linhas/permissão; não apague para esconder o problema |

Para conferir cabeçalho, use um editor de texto: `id,descricao,categoria,prioridade,status`.
Uma planilha pode alterar IDs, acentos, delimitadores ou valores. Não edite o CSV
com o aplicativo aberto; não execute dois servidores no mesmo arquivo. Para
isolar um problema sem sobrescrever tarefas, inicie com
`python -m sos_ml.rotina.app --csv data/rotina/diagnostico.csv`.

Se pedir ajuda, envie somente comando, mensagem e exemplo fictício que reproduza
o problema. Não envie seu CSV pessoal, credenciais ou conteúdo de terceiros.
