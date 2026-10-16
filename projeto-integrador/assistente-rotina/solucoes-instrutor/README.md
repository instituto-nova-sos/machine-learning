# Orientações e soluções do instrutor

[← Exercícios](../exercicios.md) · [Guia](../README.md)

## Preparação e acessibilidade

Antes das duas aulas, teste os comandos do guia na máquina da turma. Tenha uma
máquina com Gradio e exemplos instalados, e opcionalmente Ollama/Qwen3 baixado.
Use CPU; não imponha GPU nem conta. Prepare o modo regras para todas as duplas e
escolha uma demonstração coletiva de LLM se a memória for limitada. Cada encontro
tem exatamente 120 minutos com pausa; use os roteiros, sem acrescentar instalações.

Não apresente o código inteiro como conteúdo a ser digitado. O percurso é ler
contrato e regras, fazer uma mudança pequena, observar persistência, contar
palavras, comparar métodos e testar falhas. `dataclass`, HTTP, arquivo temporário
e callbacks são infraestrutura fornecida: explique suas responsabilidades sem
exigir domínio desses mecanismos em quatro horas. O aluno deve conseguir seguir
uma entrada até a saída e explicar a sua modificação.

## Respostas do nível 1

1. Finanças/alta e estudos/média. Sem `hoje`, a primeira recebe média. Nenhuma
   urgência é inferida de um calendário real.
2. Análise é uma sugestão sem efeito em disco. Confirmar chama `TaskStore.add`;
   CSV guarda ID, descrição, categoria, prioridade e status.
3. `[1,1]` e `[0,2]`. Contagens perdem ordem, negação e várias relações semânticas.
4. A regra encontra a palavra urgente sem interpretar negação. A pessoa corrige
   a tabela; o código não aprende com essa correção automaticamente.
5. `saúde` viola a lista fechada e é recusada. Uma tarefa inventada pode passar
   na estrutura, mas deve ser removida pela pessoa. JSON válido não é verdade.

## Solução de personalização e avaliação

Uma solução mínima acrescenta `redação` à tupla de estudos, mantém o contrato e
adiciona o teste proposto. Antes: outros; depois: estudos. O ML só muda se mudar
seu treino, e o LLM usa pesos externos pré-treinados. Não confundir regra nova
com ajuste de parâmetros. Na rubrica, valorize essa explicação tanto quanto o clique.

Amostras de teste são pequenas e inventadas. O baseline de maioria empata nas
sete categorias, e uma escolha constante acerta 2 de 14. Execute o avaliador para
observar erros desta versão; não use o número como evidência geral. Alterações
didáticas no conjunto de treino exigem reinício e podem mudar previsões. Se os
alunos ajustarem exemplos após consultar o teste, expliquem que virou conjunto
de desenvolvimento e reservem outro teste final antes de alegar generalização.

## Demonstração de contratos e retorno às regras

Os testes em `tests/test_rotina.py` usam respostas HTTP simuladas. Mostre três
cenários: servidor ausente, JSON com categoria proibida e JSON com tarefa válida.
Nos dois primeiros, o serviço retorna regras com aviso; no último, o resultado
segue para revisão. Nenhum teste desse arquivo mede qualidade de geração Qwen3.
Para ver os cenários: `python -m pytest tests/test_rotina.py -k ollama -v`.

O teste `tests/test_rotina_ui.py` percorre eventos registrados no Gradio, sem
abrir rede. A revisão aceita alterações humanas e rejeita categoria inválida;
salvar exige análise anterior, e sucesso limpa a revisão. Filtros preservam IDs.
A nova conversa (ou limpar no chat) descarta só sugestões e histórico da sessão,
sem apagar o CSV. Teste pelo navegador também: chamadas diretas não exercitam
renderização HTML. O CSS fica separado do Python para personalização visual.

## Fechamento em 20 minutos

Com até dez duplas, use dois minutos por dupla; acima disso forme grupos paralelos.
Exija um exemplo com erro conhecido e revisão, não só casos que deram certo.
Avalie o benefício observado, personalização, funcionamento, testes e limites.
Use a rubrica do assistente, distinta da rubrica matemática do projeto de sensores.
Ollama não é exigência de hardware para obter a pontuação completa.

Limites que precisam aparecer: CSV sem criptografia ou coordenação multiprocesso,
regras sem negação, classificador com poucos dados e LLM sujeito a invenção.
Tarefas de finanças e voluntariado devem usar exemplos fictícios; não exponha
clientes, beneficiários ou contas reais em demonstrações.
