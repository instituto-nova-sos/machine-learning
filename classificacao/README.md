# Classificação: do sensor à decisão

[← Otimização e gradiente](../otimizacao-e-gradiente/README.md) · [Aula principal](../README.md)

Imagine que uma equipe de manutenção recebe duas medidas de um equipamento: temperatura e
vibração. Queremos estimar se ele apresentará falha nas próximas 24 horas. Nesta prática, todos
os equipamentos, medidas e desfechos são **sintéticos**. O objetivo é entender o cálculo de um
classificador; as previsões não orientam manutenção nem decisões de segurança.

**Pré-requisitos:** produto escalar, derivadas, gradiente descendente e separação treino/teste.
Ao terminar, você deverá calcular uma probabilidade com sigmoid, convertê-la em decisão por
limiar, executar uma atualização de regressão logística e comparar três implementações.

## Roteiro de estudo

1. [Conceitos](conceitos.md): classe, escore, probabilidade, limiar e fronteira.
2. [Matemática passo a passo](matematica.md): entropia cruzada, derivadas e primeira atualização.
3. [Prática executável](pratica.md): Python puro → NumPy → Scikit-learn, com dados de equipamento.
4. [Exercícios](exercicios.md): fundamentos, aplicação e desafio.
5. [Referências verificadas](referencias.md): documentação oficial para aprofundar.

As [soluções do instrutor](solucoes-instrutor/README.md) ficam separadas. Tente os exercícios
antes de consultá-las.

## O que já funciona

Na raiz, com o ambiente do projeto ativado:

```bash
python -m sos_ml.classify --plot artifacts/classificacao_e_perda.png
```

O comando gera 400 observações, separa 320 para treino e 80 para teste, ajusta a escala só no
treino e compara os três modelos. Também mostra uma nova inferência e grava um gráfico da perda
e das fronteiras usando pontos de treino. Todo o processo executa em CPU, sem downloads.

Este módulo introduz acertos, falsos alertas, falhas não detectadas e um baseline. A próxima
fase, **avaliação de modelos**, aprofundará matriz de confusão, precisão, revocação, F1 e
desbalanceamento. Persistência do classificador e aplicação operacional ainda não fazem parte
desta etapa. O comando existente de previsão imobiliária continua sendo específico de regressão.
