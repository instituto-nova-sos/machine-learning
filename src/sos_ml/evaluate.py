"""Avaliação do artefato em uma partição determinística preservada.

A mesma semente e regra usadas no treinamento recuperam os índices de teste sem armazená-los no
artefato. Essa escolha é suficiente para o experimento controlado, mas um sistema real deveria
versionar explicitamente dados e partições.
"""

import argparse
from pathlib import Path

from .data import load_housing_data
from .from_scratch.data_split import train_test_indices
from .from_scratch.losses import mean_squared_error
from .model_io import load_model


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser dos caminhos de dataset e modelo."""
    parser = argparse.ArgumentParser(description="Avalia o modelo em 20% dos dados.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Calcula e apresenta RMSE nos exemplos excluídos do treinamento.

    Args:
        argv: Argumentos da CLI sem o nome do programa, ou ``None`` para usar ``sys.argv``.

    Returns:
        Zero após uma avaliação bem-sucedida.

    Note:
        A função assume que dataset, regra de partição e semente são os mesmos do treinamento. Ela
        mede desempenho preditivo neste teste sintético, não validade para o mercado real.
    """
    args = build_parser().parse_args(argv)
    frame = load_housing_data(args.data)

    # Descartamos os índices de treino e selecionamos somente a partição nunca usada nos gradientes.
    _, test_indices = train_test_indices(len(frame))
    test = frame.iloc[test_indices]
    model = load_model(args.model)
    predictions = model.predict(test["area_m2"].tolist())
    mse = mean_squared_error(test["preco_brl"].tolist(), predictions)

    # A raiz devolve a métrica à unidade do alvo: sqrt(BRL²) = BRL. Ainda assim, RMSE não é o erro
    # de cada exemplo; resíduos maiores recebem peso quadrático no resumo.
    print(f"Avaliação em {len(test)} exemplos preservados")
    print(f"RMSE: {mse**0.5:.2f} BRL")
    print("A métrica não prova utilidade, causalidade, justiça nem desempenho futuro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
