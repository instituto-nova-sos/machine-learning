# Da sigmoid à atualização dos pesos

[← Conceitos](conceitos.md) · [Índice](README.md) · [Prática →](pratica.md)

## Uma previsão com duas entradas

Considere estatísticas de treino fictícias: média de temperatura 70 °C e desvio 10 °C;
média de vibração 4 mm/s e desvio 2 mm/s. Uma medida de 80 °C e 5 mm/s torna-se:

```text
x₁ = (80 - 70) / 10 = 1
x₂ = (5 - 4) / 2 = 0,5
```

Esses valores não têm unidade: indicam quantos desvios cada medida está acima da média.
Com pesos escolhidos só para este exemplo `w=[0,8; 0,4]` e viés `b=-0,2`:

```text
z = 0,8 × 1 + 0,4 × 0,5 - 0,2 = 0,8
p = 1 / (1 + exp(-0,8)) ≈ 0,689974
```

Com limiar 0,5 a classe é 1; com limiar 0,7 é 0. As médias, desvios e pesos deste cálculo
não são a saída esperada do treinamento: são números pequenos para praticar.

## Uma perda para respostas binárias

Queremos penalizar probabilidade pequena quando o alvo é 1 e probabilidade grande quando é 0.
A entropia cruzada binária usa logaritmo natural:

```text
ℓ(y, p) = -[y log(p) + (1-y) log(1-p)]
J = (1/n) Σᵢ ℓ(yᵢ, pᵢ)
```

Se `y=1`, sobra `-log(p)`; se `y=0`, sobra `-log(1-p)`. Para `p=0,689974`, as perdas
são aproximadamente 0,371101 e 1,171101, respectivamente. O mesmo número estimado é bom
para um desfecho e ruim para o outro. Com `p=0,5`, ambas valem `log(2) ≈ 0,693147`.
A perda é adimensional: não é reais, porcentagem de erros nem quantidade de falhas.

No código, calcular `log(p)` diretamente seria frágil quando o computador arredonda `p`
para zero. Por isso usamos a forma equivalente baseada no escore:

```text
ℓ(y, z) = max(z, 0) - y z + log(1 + exp(-|z|))
```

A exponencial nunca recebe um número positivo nessa expressão. Python usa `log1p` para
calcular o último logaritmo com precisão; NumPy usa `logaddexp`. Não recortamos a perda de
uma previsão confiante e errada: `z=1000, y=0` continua tendo perda próxima de 1000.

## Por que o gradiente contém p-y?

Para valores não saturados, podemos acompanhar a regra da cadeia:

```text
dp/dz = p(1-p)
dℓ/dp = -y/p + (1-y)/(1-p)
dℓ/dz = [-y/p + (1-y)/(1-p)] × p(1-p) = p-y
dz/dwⱼ = xⱼ; dz/db = 1

dJ/dwⱼ = (1/n) Σᵢ (pᵢ-yᵢ)xᵢⱼ
dJ/db  = (1/n) Σᵢ (pᵢ-yᵢ)
```

`i` identifica o exemplo e `j` o atributo. O cancelamento é algébrico; o código calcula
diretamente `p-y`, evitando divisões por probabilidades extremas. Esses gradientes são da
entropia cruzada composta com a sigmoid, e diferem dos gradientes do MSE.

## A primeira atualização, inteiramente à mão

Reduza a um atributo padronizado para conferir cada operação:

| Exemplo | x | y | z inicial | p inicial | p-y | (p-y)x |
|---|---:|---:|---:|---:|---:|---:|
| A | -1 | 0 | 0 | 0,5 | 0,5 | -0,5 |
| B | 1 | 1 | 0 | 0,5 | -0,5 | -0,5 |

Começamos com `w=0, b=0`. Logo `dw=-0,5`, `db=0`. Com taxa `η=0,1`:

```text
w_novo = 0 - 0,1 × (-0,5) = 0,05
b_novo = 0 - 0,1 × 0 = 0
```

As novas probabilidades são aproximadamente 0,487503 e 0,512497; a perda média cai de
0,693147 para 0,668460. Agora execute esse passo na prática e compare.

Em lote, calcule todos os gradientes com os mesmos parâmetros antes de atualizá-los.
Não use o peso novo para calcular o gradiente do viés da mesma atualização.

## Da soma ao produto matricial

| Objeto | Forma | Significado |
|---|---|---|
| X | (n, d) | n exemplos, d atributos |
| w | (d,) | um peso por atributo |
| z, p, y | (n,) | um valor por exemplo |
| X.T | (d, n) | matriz transposta |
| dw | (d,) | uma derivada por peso |

```python
scores = X @ weights + bias
errors = probabilities - targets
weight_gradients = X.T @ errors / len(X)
bias_gradient = errors.mean()
```

O operador `@` acumula os mesmos produtos que os laços de Python. Mantenha `y` como
vetor `(n,)`: um alvo `(n,1)` pode gerar por broadcasting uma matriz `(n,n)` de erros
com significado errado.

Dados perfeitamente separáveis podem fazer os pesos crescerem sem limite enquanto a perda
diminui, pois não há regularização neste módulo. O projeto usa sorteio dos desfechos para
permitir sobreposição. Taxa e épocas não garantem convergência para qualquer dado. Verificamos
os gradientes por diferenças finitas e comparamos soluções numéricas com tolerância.
