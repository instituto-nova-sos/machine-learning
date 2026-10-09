"""Treino CPU, persistência e inferência em processo separado para estudar PyTorch."""

import argparse
from pathlib import Path

from .equipment_data import transform_equipment_features
from .equipment_io import load_equipment_model, save_equipment_model
from .neural_training import train_equipment


def main(argv: list[str] | None = None) -> int:
    """Treina ou apenas recarrega: o modo inferir não gera dados nem chama fit.

    Arquivos padrão são relativos à raiz. Para instalar biblioteca opcional,
    siga pytorch-na-pratica; não importa torch no caminho de treinamento NumPy.
    """
    import torch

    from .torch_models.equipment import EquipmentMLP, load_state_dict, save_state_dict

    parser = argparse.ArgumentParser(description="PyTorch didático em CPU.")
    parser.add_argument("mode", choices=["treinar", "inferir"])
    parser.add_argument("--model", type=Path, default=Path("artifacts/equipamento_torch.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/equipamento_torch.json"))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.mode == "treinar":
        result = train_equipment("torch")
        model = EquipmentMLP()
        model.copy_from_numpy(result.artifact.model)  # Estado restaurado pela validação.
        save_state_dict(model, args.model)
        save_equipment_model(result.artifact, args.metadata)
        print(f"Melhor época: {result.best_epoch}; teste final: {result.test_metrics}")
        print(f"Baseline: {result.baseline_metrics}")
        print(f"Perda treino: {result.training_loss[0]:.6f} → {result.training_loss[-1]:.6f}")
    else:
        model = load_state_dict(args.model)
        metadata = load_equipment_model(args.metadata)
        # Compare os pesos dos dois arquivos: evita combinação acidental de versões.
        import numpy as np

        from .from_scratch.backpropagation import parameters

        if any(not np.array_equal(a, b) for a, b in zip(
            parameters(model.to_numpy()), parameters(metadata.model), strict=True
        )):
            parser.error("Pesos e metadados não pertencem ao mesmo artefato.")
        X = transform_equipment_features([[80, 5]], metadata.scalers)
        print("Inferência recarregada (80 °C, 5 mm/s):", model.predict_proba(X)[0])
    print("Dados sintéticos: nenhuma saída autoriza operação de equipamento real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
