# Prática

[← Matemática](matematica.md) · [Índice do módulo](README.md) · [Exercícios →](exercicios.md)

Para `J(x)=(x-3)²`, `J'(x)=2(x-3)`. Compare:

```python
from sos_ml.from_scratch.gradient_descent import minimize_scalar

for rate in (0.05, 0.5, 1.1):
    path = minimize_scalar(lambda x: 2 * (x - 3), 10, learning_rate=rate, iterations=12)
    print(rate, path)
```

Desenhe `J(x)` e os pontos do caminho. Título: “Movimento dos parâmetros na função de custo”;
eixos: “Parâmetro x” e “Custo J(x)”. Para regressão, use `visualizar_regressao.py`: a primeira
figura mostra ajuste e a segunda, perda por época. Experimente apenas após guardar a configuração
que converge.
