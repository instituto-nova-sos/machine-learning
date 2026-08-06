"""CLI de inferência local."""

import argparse
from pathlib import Path

from .model_io import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estima preço a partir da área.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--area", type=float, required=True, help="Área positiva em m².")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.area <= 0:
        build_parser().error("--area deve ser positiva.")
    prediction = load_model(args.model).predict_one(args.area)
    print(f"Estimativa didática: R$ {prediction:,.2f}")
    print("Não use esta saída como avaliação real de imóvel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
