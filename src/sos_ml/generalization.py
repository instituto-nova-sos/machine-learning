"""Experimento deliberado de sobreajuste: capacidade maior pode memorizar ruído.

Uma árvore divide o espaço em regiões por comparações sucessivas. Aumentar a
profundidade permite regiões menores. Não substitui a logística canônica: é um
contraste que torna a capacidade visível no mesmo mundo sintético.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .equipment_data import generate_equipment_data
from .from_scratch.data_split import train_validation_test_indices


def experiment() -> tuple[list[int], list[float], list[float], float]:
    """Compara profundidades 1–12 na validação e testa somente a escolhida.

    A árvore não exige padronização. Sementes 42/17/23 fixam dados, partições e
    desempates; não são escolhidas por métricas de teste. Retorna profundidades,
    erros de treino/validação e erro de teste da configuração escolhida.
    """
    features, targets = generate_equipment_data()
    X, y = np.asarray(features), np.asarray(targets)
    train, validation, test = train_validation_test_indices(len(y))
    depths = list(range(1, 13))
    training_errors, validation_errors = [], []
    models = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=23)
        model.fit(X[train], y[train])
        models.append(model)
        training_errors.append(float(np.mean(model.predict(X[train]) != y[train])))
        validation_errors.append(float(np.mean(model.predict(X[validation]) != y[validation])))
    # Em empate, a menor profundidade aparece primeiro. Não consultamos o teste aqui.
    chosen = min(range(len(depths)), key=lambda i: validation_errors[i])
    test_error = float(np.mean(models[chosen].predict(X[test]) != y[test]))
    return depths, training_errors, validation_errors, test_error


def main(argv: list[str] | None = None) -> int:
    """Imprime números e opcionalmente salva PNG de treino versus validação."""
    parser = argparse.ArgumentParser(description="Observe sobreajuste com dados sintéticos.")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)
    depths, train, validation, test = experiment()
    for depth, a, b in zip(depths, train, validation, strict=True):
        print(f"Profundidade {depth:2}: erro treino={a:.4f}; validação={b:.4f}")
    chosen = min(range(len(depths)), key=lambda i: validation[i])
    print(f"Escolha pela validação: {depths[chosen]}; erro final de teste={test:.4f}")
    if args.plot:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.plot(depths, train, "o-", label="Treino")
        axis.plot(depths, validation, "o-", label="Validação")
        axis.set(xlabel="Profundidade máxima", ylabel="Fração de erros",
                 title="Capacidade e sobreajuste — equipamentos sintéticos")
        axis.legend()
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot)
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
