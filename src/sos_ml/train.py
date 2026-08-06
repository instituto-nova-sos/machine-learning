"""CLI de treinamento da regressão linear didática."""

import argparse
from pathlib import Path

from .data import load_housing_data
from .from_scratch.data_split import train_test_indices
from .from_scratch.linear_regression import LinearRegression1D
from .from_scratch.scaling import StandardScaler1D
from .model_io import save_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina regressão linear área→preço em CPU.")
    parser.add_argument("--data", type=Path, required=True, help="CSV com area_m2 e preco_brl.")
    parser.add_argument("--output", type=Path, required=True, help="Caminho do artefato JSON.")
    parser.add_argument(
        "--epochs", type=int, default=2_000, help="Número de épocas (padrão: 2000)."
    )
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Taxa de aprendizado.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_housing_data(args.data)
    train_indices, _ = train_test_indices(len(frame))
    train = frame.iloc[train_indices]
    train_features = train["area_m2"].tolist()
    targets = train["preco_brl"].tolist()
    scaler = StandardScaler1D.fit(train_features)
    scaled_model = LinearRegression1D()
    history = scaled_model.fit(
        scaler.transform(train_features),
        targets,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    # Converte y = ws * ((x-média)/escala) + bs para y = w*x + b.
    model = LinearRegression1D(
        weight=scaled_model.weight / scaler.scale,
        bias=scaled_model.bias - scaled_model.weight * scaler.mean / scaler.scale,
    )
    save_model(model, args.output)
    print(f"Treinamento concluído em {len(train)} exemplos. MSE final: {history[-1]:.2f} BRL²")
    print(f"Parâmetros: peso={model.weight:.2f}, viés={model.bias:.2f}")
    print(f"Artefato salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
