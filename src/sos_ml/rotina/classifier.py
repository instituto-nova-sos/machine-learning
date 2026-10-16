"""Classificação supervisionada pequena: texto → contagens → categoria.

Os exemplos são inventados para ensino; não representam pessoas ou a diversidade
linguística do Brasil. O classificador não extrai tarefas nem decide prioridades.
"""

import csv
from functools import lru_cache
from importlib.resources import files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sos_ml.rotina.domain import CATEGORIES, clean_text


def load_examples(split: str) -> tuple[list[str], list[str]]:
    """Lê somente treino ou teste do recurso CSV instalado junto com o pacote.

    As partições foram definidas antes do ajuste. O vocabulário é aprendido apenas
    no treino; o arquivo de tarefas pessoais nunca alimenta o modelo implicitamente.
    """
    if split not in {"treino", "teste"}:
        raise ValueError("Escolha treino ou teste.")
    with files("sos_ml.rotina").joinpath("exemplos.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["categoria"] not in CATEGORIES for row in rows):
        raise ValueError("Há categorias inválidas nos exemplos.")
    selected = [row for row in rows if row["particao"] == split]
    return [row["texto"] for row in selected], [row["categoria"] for row in selected]


@lru_cache(maxsize=1)
def train_classifier() -> Pipeline:
    """Ajusta uma vez por processo um modelo Naive Bayes com contagens de palavras.

    Para cada categoria c, o modelo combina log P(c) e as contagens x_j vezes
    log P(palavra_j|c). A suavização alpha=1 evita probabilidade zero para palavras
    conhecidas que não aparecem numa categoria. Independência condicional é uma
    hipótese simplificadora. Pipeline mantém representação e estimador juntos.
    Reinicie o aplicativo após editar o CSV: o cache evita treinar a cada clique.
    """
    texts, labels = load_examples("treino")
    model = Pipeline(
        [
            ("palavras", CountVectorizer(strip_accents="unicode", lowercase=True)),
            ("categoria", MultinomialNB(alpha=1.0)),
        ]
    )
    model.fit(texts, labels)
    return model


def category_by_ml(text: str) -> str:
    """Prediz uma categoria; vocabulário totalmente desconhecido retorna outros.

    Não interpretamos a maior probabilidade como garantia de acerto. Mesmo uma
    palavra conhecida num texto ambíguo pode provocar uma sugestão inadequada.
    """
    text = clean_text(text)
    model = train_classifier()
    counts = model.named_steps["palavras"].transform([text])
    if counts.nnz == 0:
        return "outros"
    return str(model.predict([text])[0])


def evaluation_rows() -> list[list[str]]:
    """Mostra cada erro no teste fixo; não usa teste para ajustar vocabulário/pesos."""
    texts, labels = load_examples("teste")
    return [[text, label, category_by_ml(text)] for text, label in zip(texts, labels, strict=True)]


if __name__ == "__main__":
    rows = evaluation_rows()
    for text, expected, predicted in rows:
        print(f"{text}: esperado={expected}; previsto={predicted}")
    correct = sum(expected == predicted for _, expected, predicted in rows)
    print(f"Acertos no teste didático: {correct}/{len(rows)}; não é evidência de uso geral.")
