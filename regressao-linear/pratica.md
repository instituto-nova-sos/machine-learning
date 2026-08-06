# Prática progressiva

[← Conceitos](conceitos.md) · [Índice do módulo](README.md) · [Exercícios →](exercicios.md)

Calcule primeiro: com `w=3`, `b=2`, `x=4`, temos `ŷ=14`. Se `y=12`, o resíduo é `-2` e o erro
quadrático é 4. Depois rode:

```bash
python scripts/gerar_dados_imoveis.py
python -m sos_ml.train --data data/processed/imoveis.csv --output artifacts/modelo_linear.json
python -m sos_ml.evaluate --data data/processed/imoveis.csv --model artifacts/modelo_linear.json
python -m sos_ml.predict --model artifacts/modelo_linear.json --area 85
python scripts/visualizar_regressao.py
```

No gráfico, observe resíduos como distâncias verticais entre pontos e reta e a perda caindo. Os
preços são sintéticos; o modelo univariado omite localização, qualidade, tempo e negociação.

Versão NumPy essencial: `predictions = x * weight + bias`; gradiente do peso:
`2 * np.mean((predictions-y) * x)`. Só depois compare com `LinearRegression().fit(X, y)`.
