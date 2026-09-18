# Dados

[← Voltar à aula principal](../README.md) · [Aprendizagem com dados](../aprendizagem-com-dados/README.md)

`raw/` é reservado a fontes originais licenciadas; nesta etapa não há fonte externa.
`processed/imoveis.csv` é gerado por `scripts/gerar_dados_imoveis.py` com semente 42.

O preço segue aproximadamente `80.000 + 3.200 × área + ruído normal`. Isso facilita observar
regressão e gradiente, mas omite localização, estado, negociação, mudanças temporais e muitos
fatores reais. Não é evidência de mercado e não serve para decisões financeiras.

O projeto de [classificação](../classificacao/README.md) gera em memória 400 observações
independentes com `sos_ml.equipment_data.generate_equipment_data`: temperatura (40–100 °C),
vibração (0,5–8 mm/s) e alvo binário de falha nas próximas 24 horas. A semente de geração é 42;
a de partição é 17. Rótulos são sorteados com probabilidade definida por uma regra logística
artificial. A saída não contém probabilidades geradoras como atributos. Faixas e prevalência
são fictícias; não representam condições operacionais de uma indústria.
