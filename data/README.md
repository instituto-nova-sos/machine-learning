# Dados

[← Voltar à aula principal](../README.md) · [Aprendizagem com dados](../aprendizagem-com-dados/README.md)

`raw/` é reservado a fontes originais licenciadas; nesta etapa não há fonte externa.
`processed/imoveis.csv` é gerado por `scripts/gerar_dados_imoveis.py` com semente 42.

O preço segue aproximadamente `80.000 + 3.200 × área + ruído normal`. Isso facilita observar
regressão e gradiente, mas omite localização, estado, negociação, mudanças temporais e muitos
fatores reais. Não é evidência de mercado e não serve para decisões financeiras.
