"""Interface de linha de comando para treinar a regressão linear didática.

O fluxo separa o teste antes de ajustar a padronização, treina somente nos exemplos permitidos e
serializa a reta nas unidades originais. Assim, o artefato pode prever diretamente a partir de m²
sem depender de um objeto de escala separado.
"""

import argparse
from pathlib import Path

from .data import load_housing_data
from .from_scratch.data_split import train_test_indices
from .from_scratch.linear_regression import LinearRegression1D
from .from_scratch.scaling import StandardScaler1D
from .model_io import save_model


def build_parser() -> argparse.ArgumentParser:
    """Define argumentos da CLI sem executar treinamento.

    Separar a construção facilita testar :func:`main` com uma lista explícita de argumentos.
    """
    parser = argparse.ArgumentParser(description="Treina regressão linear área→preço em CPU.")
    parser.add_argument("--data", type=Path, required=True, help="CSV com area_m2 e preco_brl.")
    parser.add_argument("--output", type=Path, required=True, help="Caminho do artefato JSON.")
    parser.add_argument(
        "--epochs", type=int, default=2_000, help="Número de épocas (padrão: 2000)."
    )
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Taxa de aprendizado.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executa carregamento, partição, treinamento e persistência do modelo.

    Args:
        argv: Argumentos sem o nome do programa. ``None`` faz ``argparse`` ler ``sys.argv``, como
            ocorre ao executar ``python -m sos_ml.train``.

    Returns:
        Código zero quando todo o fluxo termina. Erros de argumentos e dados são apresentados pelas
        exceções apropriadas em vez de serem convertidos silenciosamente em sucesso.
    """
    args = build_parser().parse_args(argv)
    frame = load_housing_data(args.data)

    # O teste é separado imediatamente e seu alvo não participa de escala, gradientes ou escolhas.
    train_indices, _ = train_test_indices(len(frame))
    train = frame.iloc[train_indices]

    # ``tolist`` entrega sequências simples às implementações manuais, tornando a fronteira entre
    # armazenamento tabular (pandas) e algoritmo educacional (Python puro) explícita.
    train_features = train["area_m2"].tolist()
    targets = train["preco_brl"].tolist()

    # Média e desvio são aprendidos exclusivamente do treino para evitar vazamento do teste.
    scaler = StandardScaler1D.fit(train_features)
    scaled_model = LinearRegression1D()
    history = scaled_model.fit(
        scaler.transform(train_features),
        targets,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    # O treinamento ocorre em x_padronizado = (x - média) / escala, mas o modelo salvo deve aceitar
    # m² diretamente. Desenvolvendo a equação:
    # y = ws * ((x - média) / escala) + bs
    #   = (ws / escala) * x + (bs - ws * média / escala).
    model = LinearRegression1D(
        weight=scaled_model.weight / scaler.scale,
        bias=scaled_model.bias - scaled_model.weight * scaler.mean / scaler.scale,
    )
    # Persistimos apenas depois da conversão, de modo que avaliação e inferência não precisem
    # reconstruir o scaler nem correr o risco de ajustá-lo novamente em dados futuros.
    save_model(model, args.output)
    print(f"Treinamento concluído em {len(train)} exemplos. MSE final: {history[-1]:.2f} BRL²")
    print(f"Parâmetros: peso={model.weight:.2f}, viés={model.bias:.2f}")
    print(f"Artefato salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
