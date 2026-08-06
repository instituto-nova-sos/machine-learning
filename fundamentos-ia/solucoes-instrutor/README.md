# Soluções comentadas — acesso do instrutor

[← Exercícios](../exercicios.md) · [Índice do módulo](../README.md)

As respostas são referências para discussão, não frases únicas a memorizar. Valorize justificativa,
limites e declaração de informação insuficiente.

## 1. Recuperação conceitual

1. IA é o campo amplo; ML é uma abordagem da IA que ajusta modelos a partir de experiência; DL é
   uma abordagem de ML baseada em múltiplas camadas de representação. A inclusão não é igualdade:
   há IA sem ML e ML sem redes profundas.
2. Um sistema de planejamento por busca ou um sistema simbólico de regras pode pertencer à IA sem
   ajustar parâmetros a partir de dados. Regras, estados e operações são especificados no software
   ou na base de conhecimento.
3. Um LLM moderno é normalmente um Transformer profundo treinado como modelo de linguagem. Modelos
   de linguagem anteriores, como n-gramas, podiam usar contagens sem DL.
4. Modelo é o sistema parametrizado; algoritmo de treino ajusta seus parâmetros; parâmetro é
   aprendido; hiperparâmetro é escolhido externamente ao ajuste.
5. Perda orienta a otimização; métrica avalia ou comunica desempenho. Um treino pode otimizar MSE
   e comunicar RMSE, por exemplo.

## 2. Eixos separados

| Caso | Paradigma | Tarefa | Arquitetura/família |
|---|---|---|---|
| CNN com imagens rotuladas | supervisionado | classificação | CNN |
| documentos sem categorias | não supervisionado | agrupamento | indeterminada |
| tokens ocultos reconstruídos | auto-supervisionado | predição/reconstrução de tokens | Transformer |
| valor monetário histórico | supervisionado | regressão | árvore |
| ações e recompensa | por reforço | aprender uma política/controle | indeterminada |

CNN descreve organização de operações; classificação descreve a tarefa. Uma CNN também pode
participar de regressão ou geração.

## 3. Comparação de arquiteturas

Uma resposta adequada deve incluir:

| Arquitetura | Estrutura | Mecanismo | Limite relevante |
|---|---|---|---|
| MLP | vetor | conexões densas | não codifica localidade ou ordem por padrão |
| CNN | grade/localidade | filtros compartilhados | hipótese local pode não servir a toda relação |
| RNN | sequência | estado recorrente | treino sequencial e dependências longas difíceis |
| Transformer | relações contextuais | self-attention e posição | atenção pode ter alto custo com sequências longas |

Não exigir que estudantes tratem essas frases como garantias universais.

## 4. Treino e inferência

No treino de consumo, históricos e alvos alimentam o algoritmo; uma perda orienta o ajuste de
pesos e viés. Na inferência, apenas atributos disponíveis naquele instante entram no modelo. O
consumo futuro real pode ser comparado depois que ocorrer, mas fornecê-lo como atributo antecipa a
resposta e causa vazamento.

## 5. Regra ou aprendizagem

Nos dois casos há software, entrada, validação e política de uso. Na regra, uma pessoa definiu
diretamente o limite. No modelo, parâmetros foram ajustados por um algoritmo usando dados. Uma
regra pode ser preferível quando representa bem o requisito, oferece comportamento previsível e
evita custo e risco desnecessários. Ambos precisam de evidência no contexto relevante.

## 6. Alegação de 92%

Uma análise forte questiona definição e legitimidade do alvo, prevalência, baseline, origem dos
rótulos, período, separação entre pessoas, leakage, intervalo de incerteza, falsos positivos,
falsos negativos, resultados por grupos, proxies sensíveis, mudança temporal, contestação,
supervisão e adequação de automatizar seleção.

Exemplo de reformulação limitada:

> No conjunto de teste descrito, preservado segundo o protocolo informado, o classificador
> reproduziu o rótulo histórico com 92% de acurácia. Esse resultado não demonstra compreensão,
> justiça, causalidade nem adequação para decidir contratações.

## 7. Síntese `T–E–P`

Verifique se tarefa, experiência e desempenho são observáveis; se a métrica corresponde ao uso; se
o baseline é explícito; e se a alternativa sem ML foi considerada. Não penalize a decisão de
interromper o projeto quando os dados, direitos envolvidos ou custos de erro não sustentarem o uso.
