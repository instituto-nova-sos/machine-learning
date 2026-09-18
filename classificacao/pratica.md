# Prática: falha de equipamento em três implementações

[← Matemática](matematica.md) · [Índice](README.md) · [Exercícios →](exercicios.md)

Execute os comandos na raiz, com a `.venv` ativada e `python -m pip install -e ".[dev]"`
já realizado. Os blocos Python podem ser colados no interpretador aberto com `python`;
cada bloco abaixo inclui seus próprios imports e dados para poder ser executado separadamente.

## 1. Calcule antes de treinar

Reproduza a primeira atualização apresentada na matemática:

```python
from sos_ml.from_scratch.activations import sigmoid, probability_to_class
from sos_ml.from_scratch.logistic_regression import LogisticRegressionBinary

print(sigmoid(0.0))  # 0.5: escore zero
print(probability_to_class(0.60, threshold=0.70))  # 0: decisão sem alterar os 60%

model = LogisticRegressionBinary(weights=[0.0])
X = [[-1.0], [1.0]]  # duas linhas, um atributo já padronizado
y = [0, 1]
print(model.gradients(X, y))  # ([-0.5], 0.0)
history = model.fit(X, y, learning_rate=0.1, epochs=1)
print(model.weights, model.bias)  # [0.05] 0.0
print(model.predict_proba(X))  # aproximadamente [0.487503, 0.512497]
print(history)  # aproximadamente [0.668460]
```

Leia [o módulo manual](../src/sos_ml/from_scratch/logistic_regression.py) nesta ordem:
`decision_function`, `predict_proba`, `loss`, `gradients`, `fit`. Acompanhe também a
[sigmoid](../src/sos_ml/from_scratch/activations.py). O método `fit` é nosso próprio laço:
você pode abrir sua implementação e identificar cada atualização.

## 2. Conheça os dados antes de ajustar pesos

```python
from sos_ml.equipment_data import generate_equipment_data, prepare_equipment_data

X, y = generate_equipment_data(size=400, seed=42)
for row, target in zip(X[:5], y[:5], strict=True):
    print(f"Temperatura: {row[0]:.1f} °C; vibração: {row[1]:.2f} mm/s; falha: {target}")

split = prepare_equipment_data(X, y, seed=17)
print(len(split.X_train), len(split.X_test))  # 320 80
print(split.scalers)  # média e desvio de cada coluna, calculados só no treino
print(split.X_train[0], split.y_train[0])
```

Cada linha representa uma observação independente. O gerador sorteia medidas e depois um
desfecho com probabilidade definida por uma regra artificial. Duas condições semelhantes
podem ter alvos diferentes. A probabilidade usada pelo gerador não é entregue ao classificador.
Os dados são criados em memória; nenhum CSV externo é necessário.

Leia [equipment_data.py](../src/sos_ml/equipment_data.py) e localize a ordem:
dividir índices → ajustar scalers no treino → transformar treino e teste. Um scaler por coluna
reutiliza a implementação univariada já estudada. Só os atributos são padronizados; `y` continua
0 ou 1. Estatísticas do teste não entram no ajuste. Em registros repetidos de máquinas reais,
essa divisão aleatória simples precisaria ser repensada para grupos e tempo.

## 3. Compare Python puro, NumPy e Scikit-learn

```python
import numpy as np

from sos_ml.equipment_data import generate_equipment_data, prepare_equipment_data
from sos_ml.from_scratch.logistic_regression import LogisticRegressionBinary
from sos_ml.from_scratch.logistic_regression_numpy import LogisticRegressionNumpy
from sos_ml.sklearn_models.logistic_regression import fit_logistic_regression

X, y = generate_equipment_data()
split = prepare_equipment_data(X, y)
manual = LogisticRegressionBinary([0.0, 0.0])
vectorized = LogisticRegressionNumpy([0.0, 0.0])
history = manual.fit(split.X_train, split.y_train, learning_rate=0.1, epochs=2000)
vectorized.fit(split.X_train, split.y_train, learning_rate=0.1, epochs=2000)
professional = fit_logistic_regression(
    np.asarray(split.X_train), np.asarray(split.y_train, dtype=np.int64)
)

# Confira o cálculo no treino antes de olhar o teste final.
print(manual.weights, manual.bias)
print(vectorized.weights, vectorized.bias)
print(professional.coef_, professional.intercept_)

# Uma nova observação não possui y conhecido: aqui fazemos somente inferência.
new_features = split.transform([[80.0, 5.0]])
print(manual.predict_proba(new_features))
print(vectorized.predict_proba(new_features))
positive_column = list(professional.classes_).index(1)
print(professional.predict_proba(np.asarray(new_features))[:, positive_column])
```

A [versão NumPy](../src/sos_ml/from_scratch/logistic_regression_numpy.py) herda o laço de
treino manual e substitui as operações centrais por álgebra matricial. Uma subclasse reutiliza
comportamentos da classe original; quando `fit` chama `self.gradients`, a versão NumPy executa
seu próprio método vetorizado. A API continua aceitando e devolvendo listas, com conversões
internas para arrays. Esse desenho favorece a comparação didática, não a velocidade máxima.

No [adaptador Scikit-learn](../src/sos_ml/sklearn_models/logistic_regression.py), `C=np.inf`
remove a penalização para comparar o mesmo objetivo das versões manuais. A biblioteca usa
L-BFGS, e não nosso gradiente descendente em lote. Resultados devem ser próximos, não
idênticos em todos os bits. A versão mínima declarada do projeto é Scikit-learn 1.4;
o registro de execução em `AGENTS.md` identifica o ambiente efetivamente validado.

Repare na diferença de APIs: nosso `predict_proba` devolve um vetor com P(classe=1);
Scikit-learn devolve uma coluna por classe. Consultar `classes_` deixa explícito o significado
da coluna escolhida. Aplicar nosso comparador de limiar garante a mesma convenção de empate.

## 4. Execute o experimento completo

```bash
python -m sos_ml.classify --plot artifacts/classificacao_e_perda.png
```

Com sementes e parâmetros padrão, a execução validada produziu aproximadamente:

```text
Treino: 320; teste preservado: 80.
Falhas no treino: 118; falhas no teste: 33.
Perda manual de treino: 0.683351 → 0.442527.
Python puro: 64/80 acertos; 6 falsos alertas; 10 falhas não detectadas (limiar 0,5).
NumPy: 64/80 acertos; 6 falsos alertas; 10 falhas não detectadas (limiar 0,5).
Scikit-learn: 64/80 acertos; 6 falsos alertas; 10 falhas não detectadas (limiar 0,5).
Baseline (sempre classe 0): 47/80 acertos.
Nova medida: 80 °C, 5 mm/s → probabilidade estimada de falha: 0.5696.
```

O primeiro valor da curva é medido **depois da primeira atualização**; antes do treino a perda
é log(2). Os pesos padronizados ficam próximos de `[1,357201; 1,333977]`, com viés `-0,890922`.
Eles não multiplicam diretamente °C ou mm/s: para prever, reutilize `split.transform`.
O módulo não exporta modelo treinado; uma nova execução refaz deterministicamente o experimento.

Os três modelos acertarem os mesmos 64 exemplos em quantidade não bastaria para provar
equivalência: por isso os testes também comparam probabilidades e gradientes. A comparação
com baseline usa a classe majoritária escolhida no treino, sem consultar os rótulos do teste.

No gráfico, à esquerda, observe a entropia cruzada de treino caindo. À direita, os símbolos
representam classes observadas no treino; as retas indicam probabilidades 0,3, 0,5 e 0,7.
Não são três modelos diferentes. O lado de maior temperatura e vibração tem maior probabilidade
neste mundo artificial, porque essa relação foi introduzida no gerador.

## 5. Faça perguntas sobre os erros

O modelo deixou de detectar dez falhas no teste. Isso seria aceitável? Os dados sintéticos não
permitem responder. Faltam custos, criticidade, frequência real, qualidade de sensores e uma
população representativa. Também não devemos usar esses números para escolher outro limiar:
essa escolha exigiria um conjunto de validação separado.

Para estudar taxas de aprendizado, use a perda **de treino** e mantenha o teste preservado.
Para estudar limiares, use a nova medida sem rótulo ou exemplos manuais. A avaliação detalhada
dos tipos de erro, do desbalanceamento e das métricas será o próximo módulo.

## 6. Verifique a implementação

```bash
python -m pytest tests/test_classification.py tests/test_equipment_data.py
python -m ruff check .
python -m mypy
```

Os testes conferem um passo à mão, uma solução analítica não separável, gradientes por
diferenças finitas, estabilidade em escores extremos, contratos de entrada, igualdade numérica
entre versões e ausência de influência dos dados de teste sobre treino e escala.
