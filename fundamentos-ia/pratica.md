# Prática: especificar o problema antes do algoritmo

[← Conceitos](conceitos.md) · [Índice do módulo](README.md) · [Exercícios →](exercicios.md)

## Objetivo

Produzir uma especificação curta que permita decidir se um problema é adequado para ML. Nesta
atividade, escolher “não usar ML” pode ser a conclusão tecnicamente correta.

## Etapa 1 — Complete o cartão do problema

Escolha um dos cenários abaixo e preencha:

| Campo | Pergunta |
|---|---|
| Decisão ou tarefa | O que precisa acontecer depois da saída? |
| Unidade de análise | Uma máquina, entrega, hora, pessoa ou documento? |
| Entrada disponível | O que existe no momento real da previsão? |
| Saída desejada | Valor, classe, grupo, texto ou ação? |
| Horizonte | Para quando a saída se refere? |
| Experiência `E` | Quais dados ou interações sustentariam o ajuste? |
| Desempenho `P` | Como comparar o sistema com um baseline? |
| Erros relevantes | Qual o custo de cada tipo de erro? |
| Pessoas afetadas | Quem recebe benefício, risco ou direito de contestação? |
| Alternativa simples | Uma regra ou mudança de processo seria suficiente? |

### Cenários

1. **Tempo de entrega:** estimar minutos até a chegada. Atrasos extremos podem importar mais que
   o erro médio.
2. **Falha de equipamento:** sinalizar necessidade de inspeção. Perder uma falha e gerar um alarme
   falso possuem custos diferentes.
3. **Organização de notícias:** reunir textos semelhantes. Um grupo encontrado não é uma categoria
   objetiva nem necessariamente verdadeira.
4. **Triagem de solicitações:** encaminhar mensagens ao setor apropriado. Dados históricos podem
   refletir encaminhamentos incorretos.
5. **Seleção de pessoas:** decidir contratação ou crédito. Questione se automatizar é apropriado,
   quais direitos estão envolvidos e se o alvo histórico é legítimo.

## Etapa 2 — Classifique em eixos separados

Não escreva apenas “é classificação”. Responda às três perguntas:

1. Qual é o **paradigma de aprendizagem**: supervisionado, não supervisionado,
   auto-supervisionado ou por reforço?
2. Qual é a **tarefa**: regressão, classificação, agrupamento, geração ou outra?
3. Qual **família de modelo** poderia ser considerada e que hipótese ela introduz?

Se não houver informação suficiente, escreva “indeterminado” e liste o que falta. Não invente uma
arquitetura apenas para preencher a resposta.

## Etapa 3 — Separe treino e inferência

Desenhe dois fluxos. Exemplo para consumo de energia:

```text
TREINO
histórico + alvos → transformação ajustada no treino → algoritmo → parâmetros → modelo

INFERÊNCIA
nova medição → mesma transformação → modelo → previsão
```

Marque em seu desenho:

- dois parâmetros possíveis;
- dois hiperparâmetros possíveis;
- a função de perda;
- uma métrica de avaliação;
- uma informação que causaria vazamento se usada no treino.

## Etapa 4 — Compare regra e modelo ajustado

Considere o alerta de temperatura:

```python
def regra(temperatura: float) -> bool:
    return temperatura >= 38.0
```

Discuta:

- quem escolheu o limite;
- se existe treinamento;
- como validar a regra;
- em que situação ajustar um modelo poderia ajudar;
- o que seria perdido em auditabilidade ou simplicidade.

Uma regra explícita também precisa de evidência. A diferença é a origem do comportamento, não a
dispensa de validação.

## Etapa 5 — Faça a recomendação

Finalize com uma das conclusões e justifique em até cinco frases:

- prosseguir para uma investigação com ML;
- começar por uma regra ou baseline simples;
- coletar ou corrigir dados antes de modelar;
- não automatizar a decisão.

Use a estrutura:

> Recomendamos ___ porque ___. A principal evidência necessária é ___. O erro mais crítico é ___.
> Antes do uso real, precisamos validar ___ e garantir ___.

## Critério de conclusão

A atividade está completa quando outra pessoa consegue distinguir, usando sua especificação, o
problema, os dados, o objetivo, o tipo de tarefa, o modo de aprendizagem e os riscos. O nome de uma
biblioteca não é parte obrigatória da resposta.
