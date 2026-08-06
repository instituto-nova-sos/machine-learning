"""Salva gráficos de ajuste e perda com rótulos em português."""

from pathlib import Path

import matplotlib.pyplot as plt

from sos_ml.data import load_housing_data
from sos_ml.from_scratch.data_split import train_test_indices
from sos_ml.from_scratch.linear_regression import LinearRegression1D
from sos_ml.from_scratch.scaling import StandardScaler1D


def main() -> None:
    frame = load_housing_data(Path("data/processed/imoveis.csv"))
    train_indices, _ = train_test_indices(len(frame))
    train = frame.iloc[train_indices]
    features = train["area_m2"].tolist()
    targets = train["preco_brl"].tolist()
    scaler = StandardScaler1D.fit(features)
    scaled_model = LinearRegression1D()
    history = scaled_model.fit(
        scaler.transform(features), targets, learning_rate=0.05, epochs=2_000
    )
    model = LinearRegression1D(
        scaled_model.weight / scaler.scale,
        scaled_model.bias - scaled_model.weight * scaler.mean / scaler.scale,
    )
    output = Path("artifacts")
    output.mkdir(exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(features, targets, s=14, alpha=0.6, label="Dados sintéticos")
    ordered = sorted(features)
    axes[0].plot(ordered, model.predict(ordered), color="tab:red", label="Modelo ajustado")
    axes[0].set(title="Preço em função da área", xlabel="Área (m²)", ylabel="Preço (R$)")
    axes[0].legend()
    axes[1].plot(history)
    axes[1].set(title="Perda durante o treinamento", xlabel="Época", ylabel="MSE (R$²)")
    figure.suptitle("Observe a aproximação da reta e a redução gradual da perda")
    figure.tight_layout()
    figure.savefig(output / "regressao_e_perda.png", dpi=150)
    print("Gráfico salvo em artifacts/regressao_e_perda.png.")


if __name__ == "__main__":
    main()
