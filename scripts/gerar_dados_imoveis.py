"""Gera o pequeno dataset sintético e determinístico do curso."""

from pathlib import Path

from sos_ml.data import generate_housing_data


def main() -> None:
    destination = Path("data/processed/imoveis.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_housing_data().to_csv(destination, index=False)
    print(f"Dataset sintético gerado em {destination} (semente 42, 120 exemplos).")
    print("Ele não representa preços reais nem deve orientar decisões financeiras.")


if __name__ == "__main__":
    main()
