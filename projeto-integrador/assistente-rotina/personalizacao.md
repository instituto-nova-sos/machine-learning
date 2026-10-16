# Aula 2 — Personalizar, testar e apresentar algo útil

[← Aula 1](construcao.md) · [Guia](README.md) · [Exercícios](exercicios.md)

**Duração total: 120 minutos, incluindo pausa.** Comece com a aplicação da aula 1
funcionando. Objetivo: adaptar um problema pequeno, testar casos normais e falhas,
e apresentar uma solução com benefício e limites claros. Não acrescente APIs,
bancos, autenticação ou treinamento de LLM nesta atividade de quatro horas.

## Roteiro com relógio

| Minutos | Atividade | Resultado esperado |
|---|---|---|
| 0–10 | Recuperar fluxo e escolher problema | uma pessoa usuária e uma necessidade concreta |
| 10–30 | Planejar três exemplos e critério de utilidade | escopo pequeno e resultados desejados no papel |
| 30–55 | Personalizar regras, exemplos ou textos | uma mudança funcional explicada |
| 55–65 | Pausa | dez minutos incluídos |
| 65–85 | Testar fluxo e entradas inválidas | evidências de acerto e de limite |
| 85–100 | Revisão entre duplas e ajustes | uma melhoria após feedback |
| 100–120 | Apresentações e reflexão | demonstração, limites e próximo passo |

Soma: 10 + 20 + 25 + 10 + 20 + 15 + 20 = **120 minutos**.
Para até dez duplas, reserve dois minutos por dupla no bloco final. Em turmas
maiores, faça apresentações simultâneas em grupos com avaliadores, mantendo os
20 minutos. Não crie uma terceira aula implícita para terminar funcionalidades.

## Escolher um recorte útil

Use categorias já existentes para caber no encontro. O domínio vem dos exemplos
e do problema escolhido; não é necessário inventar um sistema de perfis.

| Aplicação pessoal | Exemplos fictícios | Personalização possível | Cuidado específico |
|---|---|---|---|
| Finanças | pagar conta; registrar despesas | acrescentar `fatura` às regras; melhorar exemplos de orçamento | não inserir extratos ou sugerir investimentos |
| Estudos | revisar conteúdo; preparar prova | acrescentar `redação`; exemplos do curso | não prometer aprendizagem só por concluir tarefa |
| Emprego | atualizar currículo; preparar entrevista | acrescentar `portfólio`; exemplos de candidatura | não enviar documentos pessoais ou candidaturas automaticamente |
| Pequeno negócio | conferir estoque; responder cliente | acrescentar `encomenda`; exemplos de pedidos | não usar dados identificáveis de clientes |
| Casa | organizar limpeza; comprar alimentos | acrescentar `faxina`; exemplos da rotina doméstica | responsabilidades são combinadas pelas pessoas |
| Voluntariado | separar doações; organizar mutirão | acrescentar `campanha`; exemplos comunitários | preservar dados de beneficiários e evitar proxies de vulnerabilidade |

Escreva: “Para ___, o aplicativo ajuda a ___; consideraremos útil quando ___”.
Exemplo: “Para uma estudante, ajuda a separar revisões de provas; é útil quando
ela consegue revisar três sugestões, salvar e encontrar as pendentes em um minuto”.
Isso mede um benefício observável, sem confundi-lo com acurácia do modelo.

## Realizar uma mudança pequena e verificável

Escolha uma mudança principal: palavra-chave, conjunto de exemplos do domínio ou
texto de orientação. Antes de editar, anote a saída atual e a saída desejada.
Edite a tupla da categoria em `rules.py` ou acrescente textos **fictícios e rotulados**
com `particao=treino` no `exemplos.csv`. Reinicie o aplicativo após a alteração.
Aumentar dados não garante melhoria: exemplos ambíguos ou desequilibrados podem
piorar sugestões. Não mude o teste fixo para esconder um erro.

Se editar somente a palavra-chave, explique por que o ML continua igual. Se
editar treino, mostre uma entrada de desenvolvimento que mudou e uma que piorou
ou permaneceu igual. Compare as duas saídas sem afirmar que uma é sempre superior.
Para estudar uma categoria nova, seria preciso mudar contrato, regras, dados e
expectativas em conjunto; deixe isso como investigação posterior, não requisito
destas duas aulas.

Não personalize removendo a validação ou salvando automaticamente a resposta do
modelo. O contrato e a revisão tornam erros visíveis e corrigíveis. Não acrescente
valores financeiros, vencimentos ou outros campos sem planejar também o CSV e os
testes; o escopo atual é descrição/categoria/prioridade/status.

## Testar e registrar evidências

Execute `python -m pytest tests/test_rotina.py tests/test_rotina_ui.py`. O módulo
de interface é ignorado explicitamente se Gradio não estiver instalado; isso não
conta como verificação da UI. Não é preciso Ollama para os testes determinísticos.

Confira manualmente e anote entrada, esperado, observado e correção:

| Caso | Resultado esperado |
|---|---|
| Três tarefas separadas por `;` | três sugestões editáveis |
| Descrição vazia | mensagem em português; nada salvo |
| Categoria editada para valor desconhecido | salvamento recusado; CSV preservado |
| Dois cliques de salvar | sem duplicar descrição já pendente |
| Filtro seguido de seleção pelo nome e conclusão | só a tarefa escolhida muda |
| Fechar e reabrir o app | estado permanece no CSV |
| Ollama ausente ou resposta inválida simulada nos testes | aviso e regras como fallback |
| `não é urgente` | alta pelas regras, erro a corrigir na revisão |

O último caso ensina a reconhecer limites. Uma solução útil permite compreender
e corrigir o erro. Teste também uma frase fora do vocabulário e uma com duas
categorias plausíveis. Não use frases do teste como material de ajuste; reserve
novas entradas finais quando comparar melhorias de modelo.

## Revisão entre duplas e apresentação

Uma dupla experimenta a solução de outra com três frases fictícias. Ela registra
um benefício, um erro e uma sugestão concreta. A dupla autora faz um ajuste de
escopo pequeno, verifica de novo apenas o comportamento afetado e prepara a fala.

A apresentação de dois minutos deve mostrar: problema/público, uma personalização,
entrada → revisão → CSV → filtro → conclusão pela seleção, diferença entre regra/ML/LLM e um
erro conhecido. Se a máquina for fraca, apresente com regras e ML; descreva onde
o LLM se encaixa e demonstre o contrato por teste. Não há desconto por não baixar
pesos ou por não ter GPU.

Entrega: código alterado, três casos com esperado/observado, uma captura com dados
fictícios e nota de limite. Não entregue `.venv`, CSV pessoal ou modelo baixado.

## Rubrica desta prática (100 pontos)

| Critério | Pontos | Evidência |
|---|---:|---|
| Utilidade e escopo | 20 | público/problema e critério observável |
| Fluxo funcional e persistência | 25 | revisão, salvar, filtro, conclusão e reinício |
| Compreensão de regras, ML e LLM | 20 | explicação correta, contagem pequena e comparação |
| Personalização | 15 | mudança com antes/depois explicado |
| Testes e validação | 10 | casos normais, inválidos e fallback |
| Limites, privacidade e apresentação | 10 | erro conhecido e demonstração sem dados pessoais |

Esta rubrica é própria do assistente de quatro horas. A rubrica do projeto de
sensores continua no projeto anterior. Acesso a Ollama, RAM extra e serviços
pagos não rendem pontuação adicional.
