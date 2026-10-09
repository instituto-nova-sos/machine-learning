# Retropropagação: de cada derivada local ao gradiente

[← Redes](../redes-neurais/README.md)

Pré-requisito: regra da cadeia e forward da MLP. Objetivo: calcular os gradientes
antes de automatizar o cálculo. Retropropagação (backpropagation) percorre operações
em sentido inverso para acumular derivadas; não atualiza parâmetros por si só.

## Uma cadeia escalar completa

Escolhemos `x=2, w=0,5, b=0,1, y=1`. São valores sem unidade para estudar derivadas.

```text
z=wx+b=1,1
 a=z²=1,21
 J=(a-y)²/2=0,02205
 dJ/da=a-y=0,21
 da/dz=2z=2,2
 dJ/dz=(dJ/da)(da/dz)=0,462
 dz/dw=x=2; dz/db=1
 ∂J/∂w=0,924; ∂J/∂b=0,462
```

Com taxa 0,1: `w_novo=0,4076`, `b_novo=0,0538`. Use ambos os gradientes anteriores
para atualizar simultaneamente. Recalcule forward: z=0,869, a=0,755161,
J≈0,029973. **O passo aumentou a perda**: gradiente correto não garante taxa adequada.
Com taxa 0,01, o passo é menor e a perda diminui. Verifique, em vez de confiar numa
frase como “descer o gradiente sempre reduz o erro”.

## Vocabulário preciso

Derivada é variação local de uma função de uma variável. Derivada parcial varia
um argumento mantendo os demais fixos. Gradiente reúne todas as parciais de uma
função escalar. Grafo computacional representa dependências entre operações.
Backpropagation aplica regra da cadeia nesse grafo. Gradiente descendente usa o
gradiente em uma regra de atualização; não são seis nomes para a mesma coisa.

## Da rede mínima ao lote NumPy

Para um lote X(n,d) e alvos y(n,1):

```text
Z₁ = XW₁+b₁                 (n,h)
H = ReLU(Z₁)                (n,h)
Z₂ = HW₂+b₂                 (n,1)
P = sigmoid(Z₂)             (n,1)
J = média BCE(y,Z₂)         escalar
D₂ = (P-y)/n                (n,1)
dW₂ = H.T @ D₂              (h,1)
db₂ = soma(D₂, eixo=0)      (1,)
D₁ = (D₂ @ W₂.T)*(Z₁>0)    (n,h)
dW₁ = X.T @ D₁              (d,h)
db₁ = soma(D₁, eixo=0)      (h,)
```

A máscara `(Z₁>0)` é a derivada local da ReLU; em zero adotamos 0. A divisão por n
já está em D₂; dividir de novo em dW faria o passo pequeno demais. A transposta
soma contribuições de **todos** os exemplos para cada conexão. Gradientes são
calculados com W₂ antigo, antes de qualquer atualização.

A BCE baseada em logits reutiliza a estabilidade da logística: não calculamos
log de probabilidades arredondadas para 0. O cancelamento da derivada deixa P−y.

## Uma rede mínima com todos os intermediários

Use a camada do módulo anterior: X=[[1,2]], W₁=[[1,−1],[2,1]], b₁=[0,1],
W₂=[[1],[−1]], b₂=[0], y=[1]. Z₁=H=[[5,2]], Z₂=[[3]], P≈[[0,952574]],
J≈0,048587. Há um exemplo, portanto dividir por n não muda os valores.

| Intermediário | Valor aproximado | Por quê |
|---|---|---|
| D₂ | [[−0,047426]] | P−y |
| dW₂ | [[−0,237129],[−0,094852]] | H.T @ D₂ |
| db₂ | [−0,047426] | soma das linhas |
| D₁ | [[−0,047426;0,047426]] | D₂@W₂.T, máscara positiva |
| dW₁ | [[−0,047426;0,047426],[−0,094852;0,094852]] | X.T@D₁ |
| db₁ | [−0,047426;0,047426] | soma das linhas |

A saída positiva e o alvo 1 tornam D₂ negativo: aumentar o logit reduz a perda
localmente. O peso negativo do segundo neurônio inverte o sinal transmitido a ele.
Isso mostra a cadeia, não uma necessidade de classificar sempre como 1.

```python
import numpy as np
from sos_ml.from_scratch.dense_layer import DenseLayer
from sos_ml.from_scratch.neural_network import BinaryMLP
from sos_ml.from_scratch.backpropagation import gradients, gradient_check

model = BinaryMLP(
    DenseLayer(np.array([[1., -1.], [2., 1.]]), np.array([0., 1.]), "relu"),
    DenseLayer(np.array([[1.], [-1.]]), np.zeros(1), "linear"),
)
X, y = [[1., 2.]], [1]
print(model.hidden.forward(X), model.logits(X), model.predict_proba(X), model.loss(X, y))
for derivative in gradients(model, X, y):
    print(derivative)
print(gradient_check(model, X, y))
```

## Prática e inspeção

```bash
python -m sos_ml.network_demo
python -m pytest tests/test_backpropagation.py
```

Leia [backpropagation.py](../src/sos_ml/from_scratch/backpropagation.py): primeiro
`scalar_example`, depois `gradients`, `gradient_check`, `fit`.
Imprima Z₁/H/Z₂/D₂ em um lote de três exemplos e siga as formas acima.

Diferenças centrais aproximam `[J(θ+ε)-J(θ−ε)]/(2ε)`. Checamos cada peso e viés,
restaurando θ. ε=1e−6 em float64 funciona nos exemplos testados; ε excessivamente
pequeno sofre cancelamento. Perto da dobra da ReLU, uma perturbação pode cruzar o
ponto não diferenciável: divergência do check ali não prova bug.
A checagem é cara (duas perdas por parâmetro); não é o algoritmo de treino.

Queda da perda em equipamentos fictícios valida o mecanismo de ajuste, não
manutenção real. Avaliação e política continuam necessárias.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: Deep Learning](../deep-learning/README.md)
