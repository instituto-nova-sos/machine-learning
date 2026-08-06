# Exercícios

[← Prática](pratica.md) · [Índice do módulo](README.md) · [Referências →](referencias.md)

## 1. Recuperação conceitual

1. Explique, sem usar apenas exemplos, por que `DL ⊂ ML ⊂ IA`.
2. Dê um exemplo de IA sem ML e identifique onde suas regras são definidas.
3. Explique por que um LLM moderno usa ML por meio de Deep Learning, mas “modelo de linguagem”
   não é sinônimo de LLM.
4. Diferencie modelo, algoritmo de treinamento, parâmetro e hiperparâmetro.
5. Diferencie função de perda e métrica, incluindo um caso em que não seriam iguais.

## 2. Não misture os eixos

Para cada caso, identifique separadamente paradigma de aprendizagem, tarefa e possível arquitetura.
Quando a informação for insuficiente, registre a ambiguidade.

1. Uma CNN recebe imagens rotuladas e estima a espécie presente.
2. Um algoritmo reúne documentos sem receber categorias.
3. Um Transformer oculta tokens do próprio texto e aprende a reconstruí-los.
4. Uma árvore usa dados históricos para estimar um valor monetário.
5. Um agente escolhe ações em um simulador e recebe recompensas.

Depois explique por que “CNN” e “classificação” não são respostas concorrentes à mesma pergunta.

## 3. Compare arquiteturas

Construa uma tabela comparando MLP, CNN, RNN e Transformer em quatro dimensões:

- estrutura dos dados que a arquitetura procura explorar;
- mecanismo característico;
- vantagem esperada;
- limitação ou custo relevante.

Não use “é melhor” sem dizer para qual problema, dados e critério.

## 4. Treino e inferência

Desenhe os dois fluxos para previsão de consumo de energia. Inclua:

- entradas e alvo do treino;
- algoritmo e função de perda;
- parâmetros produzidos;
- entrada disponível na inferência;
- previsão;
- momento em que o valor real poderia ser usado para avaliação.

Explique por que fornecer o consumo futuro real como atributo seria vazamento.

## 5. Regra ou aprendizagem?

Compare os sistemas:

```python
alerta_por_regra = temperatura >= 38.0
alerta_por_modelo = modelo.predict(atributos)
```

Responda:

1. O que precisa ser programado nos dois casos?
2. Onde está o comportamento ajustado no segundo caso?
3. Que evidência seria necessária para validar cada sistema?
4. Em que condições a regra seria preferível?

## 6. Analise uma alegação

Uma empresa afirma que seu sistema “entende candidatos” porque alcançou 92% de acurácia.

Liste pelo menos dez informações ausentes, cobrindo:

- definição do alvo e unidade de análise;
- origem e período dos dados;
- prevalência e baseline;
- partições e risco de vazamento;
- falsos positivos e falsos negativos;
- resultados por grupos relevantes;
- proxies e qualidade dos rótulos;
- possibilidade de contestação e supervisão;
- mudança de distribuição;
- adequação de automatizar a decisão.

Reescreva a alegação de maneira limitada ao que a evidência realmente permitiria afirmar.

## 7. Desafio de síntese

Escolha um problema real e escreva uma ficha `T–E–P`:

- tarefa `T` precisa;
- experiência `E` disponível;
- desempenho `P` e baseline;
- paradigma de aprendizagem;
- tipo de tarefa;
- família de modelo inicial;
- alternativa sem ML;
- principal risco e critério para interromper o projeto.

Finalize decidindo se recomendaria experimentar ML. A decisão “não” é válida quando justificada.

Soluções comentadas: [solucoes-instrutor](solucoes-instrutor/README.md).
