"""Avaliação final da logística anterior com limiar fixado previamente em 0,5."""

from .equipment_data import generate_equipment_data, prepare_equipment_data
from .from_scratch.logistic_regression_numpy import LogisticRegressionNumpy
from .from_scratch.metrics import confusion_counts, majority_class


def main() -> int:
    """Treina em CPU, mostra métricas e baseline; não seleciona pelo teste."""
    X, y = generate_equipment_data()
    split = prepare_equipment_data(X, y)
    model = LogisticRegressionNumpy([0.0, 0.0])
    model.fit(split.X_train, split.y_train, epochs=2000)
    for name, labels in (
        ("Logística", model.predict(split.X_test)),
        ("Baseline", [majority_class(split.y_train)] * len(split.y_test)),
    ):
        counts = confusion_counts(split.y_test, labels)
        print(f"{name}: matriz (alvo × previsão) {counts.matrix}; {counts.metrics()}")
    print("Dados sintéticos. As métricas não autorizam decisões industriais.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
