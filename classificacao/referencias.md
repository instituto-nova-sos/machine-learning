# Referências verificadas

[← Exercícios](exercicios.md) · [Índice](README.md)

Documentação primária consultada em 18 de setembro de 2026. Os links `stable` acompanham
as versões publicadas; o ambiente efetivamente executado está registrado em `AGENTS.md`.

- [Scikit-learn — regressão logística](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression):
  objetivo, regularização e opções de solução. Leia depois de derivar o caso binário manual.
- [Scikit-learn — API LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html):
  parâmetros do estimador, `classes_` e formato das probabilidades. A documentação atual
  indica `C=np.inf` para ajuste sem penalização; essa é a configuração da comparação.
- [NumPy — logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html):
  operação usada na implementação vetorizada para calcular logaritmos de somas de exponenciais.
- [Scikit-learn — prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage):
  por que dividir dados antes de aprender transformações e preservar o teste durante o ajuste.

Os coeficientes, unidades, faixas e horizonte do gerador são decisões didáticas deste repositório.
Nenhuma dessas referências fornece evidência de validade industrial para o dataset artificial.
