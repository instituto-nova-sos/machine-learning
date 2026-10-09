# Redes neurais: formas antes de treinamento

[← Neurônio](../neuronio-artificial/README.md)

Pré-requisito: neurônio e matrizes. Objetivo: acompanhar cada eixo de uma MLP
(perceptron multicamada, multilayer perceptron), inicialmente só no cálculo de saída.

## De vários neurônios a uma matriz

Uma camada densa conecta cada entrada a cada saída. Nossa convenção é guardar pesos
por colunas: `W[j,k]` conecta atributo j ao neurônio k. Para n exemplos, d atributos
 e h saídas, `X(n,d) @ W(d,h) + b(h,) = Z(n,h)`; ativação produz `A(n,h)`.
O produto soma no eixo d, não no eixo dos exemplos. O viés se repete por linha.

```text
X=[1,2], W=[[1,-1],[2,1]], b=[0,1]
z₁=1×1+2×2+0=5; z₂=1×(-1)+2×1+1=2
A=ReLU(Z)=[5,2]
```

A próxima camada recebe A como entrada, não os sensores originais.
`W₂(h,1)` e `b₂(1,)` produzem um logit por linha `(n,1)`.
Com W₂=[[1],[-1]] e b₂=[0], o logit do exemplo é 3 e p≈0,952574.
É uma conta de pesos escolhidos, não modelo industrial treinado.

## Propagação para frente (forward propagation)

É executar as funções na ordem entrada → representações → saída. Não calcula
sozinho gradientes nem atualiza pesos. Empilhar linearidades apenas colapsaria a
uma função afim; a ReLU intermediária permite mudar relações por região.
A rede mínima tem duas transformações aprendidas, não pretende ser modelo grande.

```python
import numpy as np
from sos_ml.from_scratch.dense_layer import DenseLayer
from sos_ml.from_scratch.neural_network import BinaryMLP

hidden = DenseLayer(np.array([[1., -1.], [2., 1.]]), np.array([0., 1.]), "relu")
output = DenseLayer(np.array([[1.], [-1.]]), np.zeros(1), "linear")
model = BinaryMLP(hidden, output)
print(hidden.forward([[1., 2.]]))  # [[5,2]]
print(model.logits([[1., 2.]]))  # [[3]]
print(model.predict_proba([[1., 2.]]))
```

Execute `python -m pytest tests/test_network.py`. Leia
[dense_layer.py](../src/sos_ml/from_scratch/dense_layer.py) e
[neural_network.py](../src/sos_ml/from_scratch/neural_network.py).
Não remova eixos com `squeeze()` indiscriminado: um lote de uma linha pode virar
escalar. Alvos da futura perda terão `(n,1)`; vetor `(n,)` pode produzir erro `(n,n)`.

Inicialização pseudoaleatória quebra simetria: unidades idênticas recebem gradientes
idênticos. A semente reproduz o experimento, não melhora qualidade nem representa
incerteza. Avaliar várias sementes não permite escolher uma pelo teste.
Representações ocultas não têm necessariamente interpretação humana direta.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: backpropagation](../backpropagation/README.md)
