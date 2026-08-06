# Prática: Python puro e NumPy

[← Matemática](matematica.md) · [Índice do módulo](README.md) · [Exercícios →](exercicios.md)

```python
import numpy as np
from sos_ml.from_scratch.vectors import manual_dot_product

x = [50.0, 2.0, 10.0]
w = [3.2, 15.0, -0.5]
print(manual_dot_product(x, w))
print(np.dot(np.array(x), np.array(w)))
```

Os resultados devem coincidir dentro de tolerância. Para desenhar vetores 2D, use setas saindo
da origem, eixos “Componente x” e “Componente y” e legenda com os vetores. Observe direção,
comprimento e como a soma fecha um paralelogramo; isso é interpretação geométrica, não requisito
para todo tensor de ML.
