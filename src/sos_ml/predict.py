"""Inferência local com um modelo já treinado e salvo.

Inferência usa parâmetros existentes para calcular uma saída; não consulta alvos, não recalcula
parâmetros e não constitui uma nova etapa de treinamento.
"""

import argparse
from pathlib import Path

from .model_io import load_model


def build_parser() -> argparse.ArgumentParser:
    """Define os argumentos necessários para uma previsão individual."""
    parser = argparse.ArgumentParser(description="Estima preço a partir da área.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--area", type=float, required=True, help="Área positiva em m².")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Valida uma área, carrega o artefato e imprime a estimativa didática.

    Args:
        argv: Argumentos sem o nome do programa, ou ``None`` para ler ``sys.argv``.

    Returns:
        Zero após a inferência. ``argparse`` encerra com código não zero quando a área é inválida.
    """
    args = build_parser().parse_args(argv)
    if args.area <= 0:
        # A reta aceitaria matematicamente zero ou negativos, mas esses valores violam o significado
        # físico escolhido para a entrada. Validar o domínio impede uma extrapolação sem sentido.
        build_parser().error("--area deve ser positiva.")
    prediction = load_model(args.model).predict_one(args.area)
    print(f"Estimativa didática: R$ {prediction:,.2f}")
    print("Não use esta saída como avaliação real de imóvel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
