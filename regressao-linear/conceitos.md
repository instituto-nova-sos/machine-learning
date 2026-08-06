# Conceitos

[← Índice do módulo](README.md) · [Prática →](pratica.md)

Para área `x`, `ŷ=wx+b`. O peso é a variação prevista no preço por unidade de área e o viés é a
previsão matemática em `x=0` — que pode estar fora do domínio e não merecer interpretação física.
O resíduo é `y-ŷ`. O erro quadrático médio é `MSE=(1/n)Σ(yᵢ-ŷᵢ)²`: não negativo, diferenciável e
sensível a erros grandes. Sua unidade é o quadrado da unidade-alvo; RMSE retorna à unidade original.

Inicializamos parâmetros, prevemos, medimos perda, calculamos gradientes e atualizamos. Repetir
reduz a perda apenas se escolhas e implementação forem adequadas. Convergência não prova que o
modelo representa causalidade ou funcionará fora da distribuição observada.

Há três níveis:

1. `from_scratch/linear_regression.py` expõe previsões, derivadas e atualizações;
2. NumPy representa lotes e executa operações vetorizadas;
3. `sklearn_models/linear_regression.py` resolve o ajuste por uma API consolidada.

Scikit-learn abstrai álgebra e estimação, mas o engenheiro ainda define problema, dados,
partições, baseline, métrica, validação e uso responsável. `fit()` correto na sintaxe ainda pode
treinar com vazamento, alvo inadequado ou amostra enviesada.
