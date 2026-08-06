# Prática: auditar o dataset sintético

[← Conceitos](conceitos.md) · [Índice do módulo](README.md) · [Exercícios →](exercicios.md)

```bash
python scripts/gerar_dados_imoveis.py
python -c "import pandas as pd; d=pd.read_csv('data/processed/imoveis.csv'); print(d.describe())"
```

Confirme colunas, unidades, ausentes, faixa e número de exemplos. Documente: geração com semente
42, relação linear artificial e ruído; ausência de fatores reais; proibição de uso financeiro.
Planeje a divisão antes de calcular parâmetros de transformação.
