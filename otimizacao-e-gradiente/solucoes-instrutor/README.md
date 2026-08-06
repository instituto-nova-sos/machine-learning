# Soluções comentadas — acesso do instrutor

[← Exercícios](../exercicios.md) · [Índice do módulo](../README.md)

Com derivada `2x`: `4→3,2→2,56`. Curva decrescente e parâmetros estabilizando sugerem
convergência; crescimento, `inf` ou alternância crescente sugerem divergência. Para checar o peso,
avalie o MSE em `w+ε` e `w-ε` mantendo o viés, divida por `2ε` e compare com `gradients` usando
tolerância. O custo é ao menos duas avaliações por parâmetro e há erro numérico.
