# Soluções comentadas — acesso do instrutor

[← Exercícios](../exercicios.md) · [Índice do módulo](../README.md)

O produto é `4+0-3=1`; a forma é `(5,3)`. Para `(4,3) @ (4,1)`, as dimensões internas 3 e 4 não
coincidem; talvez a intenção seja transpor a primeira ou usar um vetor de pesos `(3,1)`. Forma
compatível não garante semântica: somar um vetor indexado por exemplo quando ele representa
atributos mistura entidades diferentes sem erro da biblioteca.
