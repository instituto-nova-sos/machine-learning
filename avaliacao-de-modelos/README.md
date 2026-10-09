# Avaliação de modelos: quem errou e com qual consequência?

[← Classificação](../classificacao/README.md) · [Trilha](../docs/jornada-de-aprendizado.md)

Pré-requisito: distinguir escore, probabilidade e classe. Aqui usamos o mesmo alvo:
falha **sintética** nas próximas 24 horas. Antes de executar, conte dez pares à mão.

## Da contagem à proporção

Positivo significa classe 1, não “resultado bom”. Verdadeiro positivo (VP) é alerta
com falha; verdadeiro negativo (VN) é ausência de alerta sem falha. Falso positivo
(FP) é falso alerta; falso negativo (FN) é falha não detectada.

| Alvo \ previsão | 0 | 1 |
|---|---:|---:|
| 0 | VN=3 | FP=1 |
| 1 | FN=2 | VP=4 |

Há dez casos: sete acertos. Definimos `n=VN+FP+FN+VP`:

- Acurácia (accuracy): `(VP+VN)/n = 7/10`.
- Precisão (precision): `VP/(VP+FP) = 4/5`. Entre os alertas, quantos eram positivos?
- Revocação ou sensibilidade (recall): `VP/(VP+FN) = 4/6`. Quantas falhas detectamos?
- Especificidade: `VN/(VN+FP) = 3/4`. Quantos negativos foram reconhecidos?
- F1: `2VP/(2VP+FP+FN) = 8/11`. É a média harmônica de precisão e revocação
  quando a expressão em termos das duas taxas está definida. Não inclui VN.

Uma razão sem denominador é indefinida. Nosso código retorna `None`, não inventa
uma taxa zero. Bibliotecas podem adotar outra convenção; registre-a ao comparar.

## Por que 90% pode ser ruim?

Cem observações têm noventa negativos e dez positivos. Sempre prever 0 acerta 90%,
mas detecta zero das dez falhas. O baseline de classe majoritária é escolhido no
**treino**, não no teste. Compare métricas por classe, prevalência e contagens,
em vez de aceitar uma acurácia isolada. Amostras pequenas também geram incerteza.

FP pode gastar tempo de inspeção; FN pode omitir um problema. Um custo ilustrativo
`C = 2FP + 20FN` daria 42 na matriz acima. Esses números são inventados, não custos
industriais. Mesmo um custo esperado pequeno não determina quem pode autorizar ações.
Métricas agregadas podem esconder diferenças entre grupos e sensores.

## Prática

```python
from sos_ml.from_scratch.metrics import confusion_counts
from sos_ml.from_scratch.activations import probability_to_class

y = [0, 0, 1, 1]
p = [0.1, 0.4, 0.6, 0.9]  # exemplo manual; não é teste do projeto
for threshold in (0.3, 0.5, 0.7):
    labels = [probability_to_class(value, threshold=threshold) for value in p]
    counts = confusion_counts(y, labels)
    print(threshold, counts.matrix, counts.metrics())
```

Diminuir o limiar amplia alertas; FP e VP não diminuem nas mesmas probabilidades.
Aumentá-lo pode omitir falhas. A probabilidade permanece fixa. Esse exemplo mostra
o mecanismo; **não** selecionamos limiar usando os 80 casos de teste conhecidos.

Execute na raiz:

```bash
python -m sos_ml.assess
python -m pytest tests/test_metrics.py
```

Leia [metrics.py](../src/sos_ml/from_scratch/metrics.py) antes de comparar com
Scikit-learn nos testes. O comando mantém limiar 0,5 previamente fixado e apresenta
avaliação final da logística anterior. Não reinterprete seu resultado como segurança real.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: generalização](../generalizacao-e-overfitting/README.md)
