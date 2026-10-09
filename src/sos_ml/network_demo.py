"""Prática CPU: imprime cadeia escalar e treina a MLP no domínio familiar."""

from .equipment_data import generate_equipment_data, prepare_equipment_data
from .from_scratch.backpropagation import fit, gradient_check, scalar_example
from .from_scratch.neural_network import BinaryMLP


def main() -> int:
    """Mostra cálculo e queda da perda sem usar teste para orientar o treino."""
    print("Cadeia escalar:", scalar_example())
    X, y = generate_equipment_data()
    split = prepare_equipment_data(X, y)
    model = BinaryMLP.initialize()
    print("Erro máximo do gradient check:", gradient_check(model, split.X_train[:3],
                                                           split.y_train[:3]))
    initial = model.loss(split.X_train, split.y_train)
    history = fit(model, split.X_train, split.y_train)
    print(f"MLP NumPy: perda de treino {initial:.6f} → {history[-1]:.6f}")
    print("Nova medida sintética:", model.predict_proba(split.transform([[80, 5]])))
    print("Queda de treino não demonstra generalização ou segurança industrial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
