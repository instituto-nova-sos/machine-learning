# Prática

[← Matemática](matematica.md) · [Índice do módulo](README.md) · [Exercícios →](exercicios.md)

Execute e compare valor analítico e aproximação:

```python
from sos_ml.from_scratch.gradient_descent import finite_difference

quadratic = lambda x: x**2
print(2 * 3.0)
print(finite_difference(quadratic, 3.0))
```

Para visualizar uma função linear e sua inclinação, desenhe `y=3x+2` para `x` entre -2 e 4 e
marque os pontos `(1,5)` e `(2,8)`. Eixos: “Entrada x” e “Saída f(x)”; legenda: “f(x)=3x+2”.
Observe que a variação vertical é 3 quando a horizontal é 1.
