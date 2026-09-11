"""Materializa em CSV o pequeno dataset sintético e determinístico do curso.

Execute este script a partir da raiz do repositório. O caminho relativo mantém comandos e materiais
iguais entre máquinas, desde que a estrutura do projeto seja preservada.
"""

from pathlib import Path

from sos_ml.data import generate_housing_data


def main() -> None:
    """Gera 120 exemplos com semente 42 e os grava sem índice artificial.

    O índice do DataFrame não é uma variável do problema; por isso ``index=False`` evita criar uma
    terceira coluna sem significado para o modelo.
    """
    destination = Path("data/processed/imoveis.csv")

    # O diretório de dados processados pode não existir em um clone recém-preparado.
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_housing_data().to_csv(destination, index=False)
    print(f"Dataset sintético gerado em {destination} (semente 42, 120 exemplos).")
    print("Ele não representa preços reais nem deve orientar decisões financeiras.")


if __name__ == "__main__":
    main()
