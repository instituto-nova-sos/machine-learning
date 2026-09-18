"""Experimento reproduzível: ``python -m sos_ml.classify`` na raiz do projeto.

Gera dados sintéticos, divide, padroniza, treina as três versões e compara saídas.
A avaliação final apenas relata resultados: não escolhe taxa, épocas ou limiar.
Opcionalmente salva um gráfico de TREINO para estudar perda e fronteira de decisão.
Não salva um modelo; a persistência de classificadores pertence a uma etapa futura.
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .equipment_data import EquipmentSplit, generate_equipment_data, prepare_equipment_data
from .from_scratch.activations import probability_to_class
from .from_scratch.logistic_regression import LogisticRegressionBinary
from .from_scratch.logistic_regression_numpy import LogisticRegressionNumpy
from .sklearn_models.logistic_regression import fit_logistic_regression


def plot_training(
    model: LogisticRegressionBinary,
    split: EquipmentSplit,
    history: list[float],
    output: Path,
) -> None:
    """Salva PNG com perda por época e fronteiras no espaço das medidas físicas.

    Só desenha pontos de treino. Contornos 0,3/0,5/0,7 ilustram decisões alternativas,
    sem selecionar um limiar pelo teste. Cria diretórios pais e substitui o PNG no
    caminho solicitado. Usa backend Agg para funcionar em CPU sem janela gráfica.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temperatures: NDArray[np.float64]
    vibrations: NDArray[np.float64]
    # A grade cobre combinações de medidas físicas; cada célula recebe uma probabilidade.
    # Achatamos as duas matrizes em 10.000 linhas com duas colunas para usar a API do modelo.
    temperatures, vibrations = np.meshgrid(np.linspace(40, 100, 100), np.linspace(0.5, 8, 100))
    grid = np.column_stack([temperatures.ravel(), vibrations.ravel()]).tolist()
    probabilities = np.asarray(model.predict_proba(split.transform(grid))).reshape(100, 100)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(range(1, len(history) + 1), history)
    axes[0].set(xlabel="Época", ylabel="Entropia cruzada média", title="Perda de treino")
    # Desfazemos x_padronizado=(x-média)/desvio somente para rotular os eixos em °C e mm/s.
    # O modelo continua recebendo entradas padronizadas, inclusive na grade acima.
    physical = np.asarray(split.X_train) * [s.scale for s in split.scalers] + [
        s.mean for s in split.scalers
    ]
    for label, marker, color in [(0, "o", "tab:blue"), (1, "x", "tab:orange")]:
        selected = np.asarray(split.y_train) == label
        axes[1].scatter(
            physical[selected, 0],
            physical[selected, 1],
            marker=marker,
            color=color,
            label=f"Classe {label} no treino",
            alpha=0.65,
        )
    contours = axes[1].contour(
        temperatures, vibrations, probabilities, levels=[0.3, 0.5, 0.7], colors="black"
    )
    axes[1].clabel(contours, fmt="p=%.1f")
    axes[1].set(xlabel="Temperatura (°C)", ylabel="Vibração (mm/s)", title="Limiares de decisão")
    axes[1].legend()
    figure.suptitle("Equipamentos fictícios — dados sintéticos, sem validade operacional")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Executa a aula com sementes, taxa, épocas e limiar previamente fixados.

    Aceita somente --plot CAMINHO opcional para salvar o gráfico. Retorna zero em
    caso de sucesso; argparse encerra com código 2 para argumentos inválidos.
    O baseline prevê a classe mais frequente no TREINO. Exibimos contagens simples
    de acertos e erros para iniciar a discussão; métricas detalhadas ficam para o
    próximo módulo. Concordância entre bibliotecas é verificação de implementação,
    não validação da aplicação industrial.
    """
    parser = argparse.ArgumentParser(description="Classificação didática de falha de equipamento.")
    parser.add_argument("--plot", type=Path, help="Caminho do PNG de treino a gerar.")
    args = parser.parse_args(argv)
    features, targets = generate_equipment_data()
    split = prepare_equipment_data(features, targets)
    print("Dataset sintético: 400 exemplos; geração com semente 42; partição com semente 17.")
    print(f"Treino: {len(split.y_train)}; teste preservado: {len(split.y_test)}.")
    print(f"Falhas no treino: {sum(split.y_train)}; falhas no teste: {sum(split.y_test)}.")
    manual = LogisticRegressionBinary([0.0, 0.0])
    vectorized = LogisticRegressionNumpy([0.0, 0.0])
    # Os hiperparâmetros são fixados antes da avaliação, iguais nas duas versões.
    history = manual.fit(split.X_train, split.y_train, learning_rate=0.1, epochs=2000)
    vectorized.fit(split.X_train, split.y_train, learning_rate=0.1, epochs=2000)
    professional = fit_logistic_regression(
        np.asarray(split.X_train), np.asarray(split.y_train, dtype=np.int64)
    )
    print(f"Perda manual de treino: {history[0]:.6f} → {history[-1]:.6f}.")
    print(f"Pesos padronizados: {manual.weights}; viés: {manual.bias:.6f}.")
    positive_column = list(professional.classes_).index(1)
    predictions = {
        "Python puro": manual.predict_proba(split.X_test),
        "NumPy": vectorized.predict_proba(split.X_test),
        "Scikit-learn": professional.predict_proba(np.asarray(split.X_test))[
            :, positive_column
        ].tolist(),
    }
    for name, probabilities in predictions.items():
        labels = [probability_to_class(p) for p in probabilities]
        hits = sum(a == b for a, b in zip(labels, split.y_test, strict=True))
        false_alarms = sum(a == 1 and b == 0 for a, b in zip(labels, split.y_test, strict=True))
        missed = sum(a == 0 and b == 1 for a, b in zip(labels, split.y_test, strict=True))
        print(
            f"{name}: {hits}/{len(labels)} acertos; "
            f"{false_alarms} falsos alertas; {missed} falhas não detectadas (limiar 0,5)."
        )
    # Somar rótulos 0/1 conta os positivos. Em empate, adotamos classe 1 explicitamente.
    majority = int(sum(split.y_train) >= len(split.y_train) / 2)
    baseline_hits = sum(y == majority for y in split.y_test)
    print(f"Baseline (sempre classe {majority}): {baseline_hits}/{len(split.y_test)} acertos.")
    # Inferência: a nova medida usa os mesmos scalers, sem fit e sem rótulo conhecido.
    probability = manual.predict_proba(split.transform([[80.0, 5.0]]))[0]
    print(f"Nova medida: 80 °C, 5 mm/s → probabilidade estimada de falha: {probability:.4f}.")
    for threshold in (0.3, 0.5, 0.7):
        label = probability_to_class(probability, threshold=threshold)
        print(f"Limiar {threshold:.1f}: classe {label}. A probabilidade permanece igual.")
    if args.plot is not None:
        plot_training(manual, split, history, args.plot)
        print(f"Gráfico de treino salvo em: {args.plot}")
    print("Dados fictícios: não use estas previsões para manutenção ou segurança de máquinas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
