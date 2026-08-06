"""CLI de avaliação em uma partição determinística preservada."""

import argparse
from pathlib import Path

from .data import load_housing_data
from .from_scratch.data_split import train_test_indices
from .from_scratch.losses import mean_squared_error
from .model_io import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia o modelo em 20% dos dados.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_housing_data(args.data)
    _, test_indices = train_test_indices(len(frame))
    test = frame.iloc[test_indices]
    model = load_model(args.model)
    predictions = model.predict(test["area_m2"].tolist())
    mse = mean_squared_error(test["preco_brl"].tolist(), predictions)
    print(f"Avaliação em {len(test)} exemplos preservados")
    print(f"RMSE: {mse**0.5:.2f} BRL")
    print("A métrica não prova utilidade, causalidade, justiça nem desempenho futuro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
