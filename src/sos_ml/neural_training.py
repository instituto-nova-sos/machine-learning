"""Treino/validação/teste da mesma rede por NumPy ou PyTorch, sem alterar a trilha anterior."""

from dataclasses import dataclass
from typing import Literal

from .equipment_data import generate_equipment_data, prepare_development_data
from .equipment_io import EquipmentArtifact
from .from_scratch.activations import probability_to_class
from .from_scratch.backpropagation import fit, parameters
from .from_scratch.metrics import confusion_counts, majority_class
from .from_scratch.neural_network import BinaryMLP


@dataclass
class TrainingResult:
    """Artefato restaurado e histórico; teste é resumo final, não curva de seleção."""

    artifact: EquipmentArtifact
    training_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    test_metrics: dict[str, float | None]
    baseline_metrics: dict[str, float | None]


def train_equipment(backend: Literal["numpy", "torch"] = "numpy", *,
                    epochs: int = 1000, patience: int = 100) -> TrainingResult:
    """Treina 240 exemplos, monitora 80 de validação e testa 80 ao final.

    Taxa 0,1, rede 2→4→1 e sementes são fixadas antes do teste. Conserva uma cópia
    do melhor estado pela BCE de validação e restaura-a. Paciência conta épocas
    consecutivas sem melhora de pelo menos 1e-8. Históricos incluem estado inicial
    na posição zero. Teste nunca entra nos gradientes ou no critério de parada.
    NumPy é o caminho base; backend torch só importa biblioteca ao ser solicitado.
    """
    if backend not in ("numpy", "torch"):
        raise ValueError("Backend deve ser numpy ou torch.")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (epochs, patience)):
        raise ValueError("Épocas e paciência devem ser inteiros positivos.")
    X, y = generate_equipment_data()
    split = prepare_development_data(X, y)
    model = BinaryMLP.initialize()
    if backend == "torch":
        import torch

        from .torch_models.equipment import EquipmentMLP, train_step

        torch.set_num_threads(1)  # Redes minúsculas não precisam de várias threads.
        torch_model = EquipmentMLP()
        optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.1)
    training = [model.loss(split.X_train, split.y_train)]
    validation = [model.loss(split.X_validation, split.y_validation)]
    best_loss, best_epoch, stale = validation[0], 0, 0
    best_parameters = [p.copy() for p in parameters(model)]
    for epoch in range(1, epochs + 1):
        if backend == "numpy":
            train_loss = fit(model, split.X_train, split.y_train, epochs=1)[0]
        else:
            train_loss = train_step(torch_model, split.X_train, split.y_train, optimizer)
            # Copiar valores para monitorar a mesma BCE manual não cria novo treino.
            model = torch_model.to_numpy()
        training.append(train_loss)
        validation.append(model.loss(split.X_validation, split.y_validation))
        if validation[-1] < best_loss - 1e-8:
            best_loss, best_epoch, stale = validation[-1], epoch, 0
            best_parameters = [p.copy() for p in parameters(model)]
        else:
            stale += 1
        if stale >= patience:
            break
    for destination, source in zip(parameters(model), best_parameters, strict=True):
        destination[:] = source
    labels = [probability_to_class(p) for p in model.predict_proba(split.X_test)]
    baseline = [majority_class(split.y_train)] * len(split.y_test)
    return TrainingResult(EquipmentArtifact(model, split.scalers), training, validation,
                          best_epoch, confusion_counts(split.y_test, labels).metrics(),
                          confusion_counts(split.y_test, baseline).metrics())
