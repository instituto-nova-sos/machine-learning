# Neurônio artificial: reconhecer o que já construímos

[← Generalização](../generalizacao-e-overfitting/README.md)

Pré-requisito: produto escalar e sigmoid. Objetivo: compor soma ponderada e ativação
sem atribuir propriedades biológicas ou consciência à operação matemática.

## Do vetor ao escalar

`z = w·x+b = Σ_j w_j x_j + b`; `a = g(z)`.
`x` é o vetor de d entradas; `w` tem d pesos na mesma ordem; `b` é o viés escalar;
`z` é a soma ponderada; `g` é a ativação; `a` é a saída. Nos sensores, padronizar
remove as unidades de °C e mm/s; não muda o significado nem a ordem das colunas.

Com `x=[1;0,5]`, `w=[0,8;0,4]`, `b=-0,2`, temos `z=0,8`.
Se g(z)=z, é a operação da regressão linear. Se g é sigmoid, `a≈0,689974`,
exatamente a probabilidade da logística. O neurônio recompõe fundamentos conhecidos.
Uma probabilidade não é uma ação e uma ReLU não produz probabilidade.

## Ativações

Sigmoid: `1/(1+exp(-z))`, derivada `a(1-a)`, saída [0,1]. Nos extremos a derivada
é pequena, dificultando transmitir gradientes por muitas composições.
ReLU: `max(0,z)`, derivada 1 em positivos e 0 em negativos. Em zero não é
 diferenciável; adotamos derivada zero como convenção computacional.
Uma unidade negativa em todos os exemplos pode receber gradiente zero e não se recuperar.
Tanh: opcional, saída em [-1,1], derivada `1-a²`; também satura.

## Por que a não linearidade importa?

Duas camadas afins sem ativação: `h=XW₁+b₁`, `z=hW₂+b₂`.
Substituindo, `z=X(W₁W₂)+(b₁W₂+b₂)`: outra transformação afim.
Mais camadas assim não criam uma fronteira curva. A ReLU/sigmoid intermediária
impede essa redução geral e permite relações diferentes entre regiões de entrada.

## Prática em Python puro antes do NumPy

```python
from sos_ml.from_scratch.neuron import Neuron

for activation in ("linear", "sigmoid", "relu"):
    neuron = Neuron([0.8, 0.4], -0.2, activation)
    print(activation, neuron.forward([1.0, 0.5]))
```

Leia [neuron.py](../src/sos_ml/from_scratch/neuron.py). Execute
`python -m pytest tests/test_neuron.py`. A próxima camada aplicará essas contas
simultaneamente a vários exemplos e saídas. Pesos aqui são escolhidos para estudar,
não evidência de comportamento de máquinas reais.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: redes neurais](../redes-neurais/README.md)
