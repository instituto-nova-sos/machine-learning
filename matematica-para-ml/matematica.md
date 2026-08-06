# Conceitos e matemática

[← Índice do módulo](README.md) · [Prática →](pratica.md)

Uma variável representa um valor; uma função associa entradas a saídas. Em `f(x)=3x+2`, `x` é
independente, `f(x)` dependente, 3 é inclinação e 2 intercepto. Graficamente é uma reta: aumentar
`x` em 1 aumenta `f(x)` em 3. Para `x=4`, `f(x)=14`. Em Python: `lambda x: 3*x+2`. Em ML, peso e
viés formam uma função-modelo.

Potência repete multiplicação (`x²`); raiz quadrada inverte o quadrado não negativo. A soma
`Σᵢ xᵢ` compacta `x₁+...+xₙ`; a média divide essa soma por `n`. A distância euclidiana em uma
dimensão é `|a-b|`; em várias, `sqrt(Σᵢ(aᵢ-bᵢ)²)`. Perdas usam essas operações.

O logaritmo `log_b(x)` responde qual expoente de `b` produz `x`; por exemplo, `log₂(8)=3`.
Logaritmos transformam produtos em somas e aparecem em perdas probabilísticas; exigem argumento
positivo.

A derivada `f'(x)=lim(h→0)[f(x+h)-f(x)]/h` mede a taxa local de mudança e a inclinação da tangente.
Para `f(x)=x²`, `f'(x)=2x`; em `x=3`, inclinação 6. Python pode aproximá-la por diferenças finitas,
mas essa aproximação custa avaliações extras e sofre com escolha de `h`.

Com várias entradas, `∂f/∂x` varia `x` mantendo as demais fixas. O gradiente
`∇f=[∂f/∂x₁,...,∂f/∂xₙ]` aponta a direção de maior aumento local; seu negativo é usado para
reduzir uma função. Em `f(w,b)=w²+b²`, `∇f=[2w,2b]`.

A regra da cadeia deriva composições: se `y=f(g(x))`, então `dy/dx=f'(g(x))g'(x)`. Para
`(3x+2)²`, em `x=1`, a derivada é `2(5)·3=30`. Backpropagation reutilizará produtos de derivadas
locais. Pontos com gradiente zero podem ser mínimos, máximos ou selas; precisam de contexto.

> **Analogia limitada:** a curva como estrada ajuda a ver inclinação, mas derivadas também se
> aplicam a espaços de milhares de parâmetros que não visualizamos como uma estrada.
