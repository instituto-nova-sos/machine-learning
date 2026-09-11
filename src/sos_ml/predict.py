"""Inferência local com um modelo já treinado e salvo.

Inferência usa parâmetros existentes para calcular uma saída; não consulta alvos, não recalcula
parâmetros e não constitui uma nova etapa de treinamento.
"""

import argparse
import math
from pathlib import Path

from .model_io import load_model


def build_parser() -> argparse.ArgumentParser:
    """Define os argumentos necessários para uma previsão individual."""
    parser = argparse.ArgumentParser(description="Estima preço a partir da área.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--area", type=float, required=True, help="Área positiva e finita em m².")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Valida uma área, carrega o artefato e imprime a estimativa didática.

    Args:
        argv: Argumentos sem o nome do programa, ou ``None`` para ler ``sys.argv``.

    Returns:
        Zero após a inferência. ``argparse`` encerra com código não zero quando a área é inválida.
    """
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.area) or args.area <= 0:
        # ``float`` aceita as grafias ``nan`` e ``inf``. Embora elas não sejam menores ou iguais a
        # zero em todas as comparações, não representam uma medida física nem uma entrada válida
        # para a regressão; por isso a finitude precisa ser verificada explicitamente.
        #
        # Esta validação rejeita entradas não físicas básicas, mas não afirma que toda área positiva
        # pertence ao domínio observado no treino. O artefato atual guarda somente peso e viés, não
        # os limites das áreas de treino; portanto valores positivos extremos ainda são
        # extrapolações matematicamente aceitas e devem ser interpretados com cautela.
        build_parser().error("--area deve ser positiva e finita.")
    prediction = load_model(args.model).predict_one(args.area)
    print(f"Estimativa didática: R$ {prediction:,.2f}")
    print("Não use esta saída como avaliação real de imóvel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
