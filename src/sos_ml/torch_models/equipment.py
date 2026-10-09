"""Mesma MLP NumPy, agora com tensores e autograd; todas as operações em CPU."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from ..from_scratch.backpropagation import parameters
from ..from_scratch.dense_layer import FloatArray
from ..from_scratch.neural_network import BinaryMLP


class EquipmentMLP(nn.Module):  # type: ignore[misc]
    """Arquitetura 2→4→1 com ReLU oculta e logit final, usando float64 em CPU.

    nn.Linear guarda pesos (saídas,entradas), transpostos em relação ao NumPy.
    Module registra subcamadas/Parameters; não define treino nem política. O
    ignore local trata imports opcionais ignorados pela configuração mypy do
    projeto, não desativa a verificação do restante deste módulo.
    """

    def __init__(self) -> None:
        """Cria camadas pequenas e copia a inicialização NumPy determinística.

        Inicializadores PyTorch consomem RNG: fork_rng restaura seu estado para
        não interferir com aplicações externas. Não precisamos de CUDA/MPS.
        """
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.hidden = nn.Linear(2, 4, dtype=torch.float64, device="cpu")
            self.output = nn.Linear(4, 1, dtype=torch.float64, device="cpu")
        self.copy_from_numpy(BinaryMLP.initialize())

    def copy_from_numpy(self, model: BinaryMLP) -> None:
        """Copia parâmetros com transpostas, mantendo o mesmo cálculo matemático."""
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(model.hidden.weights.T.copy()))
            self.hidden.bias.copy_(torch.from_numpy(model.hidden.bias.copy()))
            self.output.weight.copy_(torch.from_numpy(model.output.weights.T.copy()))
            self.output.bias.copy_(torch.from_numpy(model.output.bias.copy()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Recebe float64 CPU (n,2); retorna logits (n,1), sem sigmoid nem limiar."""
        if (inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != 2
                or inputs.dtype != torch.float64 or inputs.device.type != "cpu"
                or not bool(torch.isfinite(inputs).all())):
            raise ValueError("Forneça tensor float64 CPU finito com forma (n,2).")
        return cast(torch.Tensor, self.output(torch.relu(self.hidden(inputs))))

    def predict_proba(self, inputs: list[list[float]]) -> list[float]:
        """Inferência sem grafo de gradientes; eval não é o mesmo que no_grad.

        Retorna n probabilidades e muda o modo para avaliação. Nosso modelo não
        usa dropout/BatchNorm, mas preservar o contrato permite estudá-los depois.
        """
        self.eval()
        with torch.no_grad():
            scores = self(torch.tensor(inputs, dtype=torch.float64, device="cpu"))
            return [float(p) for p in torch.sigmoid(scores)[:, 0].tolist()]

    def to_numpy(self) -> BinaryMLP:
        """Exporta os mesmos parâmetros para inferência NumPy local e comparação.

        detach remove vínculo com o grafo; cpu explicita dispositivo; numpy cria
        representação numérica. A cópia impede compartilhar armazenamento mutável.
        """
        result = BinaryMLP.initialize()
        arrays: list[FloatArray] = [
            np.array(self.hidden.weight.detach().cpu().numpy().T, copy=True),
            np.array(self.hidden.bias.detach().cpu().numpy(), copy=True),
            np.array(self.output.weight.detach().cpu().numpy().T, copy=True),
            np.array(self.output.bias.detach().cpu().numpy(), copy=True),
        ]
        for destination, source in zip(parameters(result), arrays, strict=True):
            destination[:] = source
        return result


def train_step(model: EquipmentMLP, X: list[list[float]], y: list[int],
               optimizer: torch.optim.Optimizer) -> float:
    """Executa uma atualização em lote e retorna a perda DEPOIS do passo.

    X(n,2), y(n,) vira tensor alvo (n,1) para coincidir com logits. A perda usa
    BCEWithLogitsLoss: sigmoid e logaritmos combinados preservam estabilidade.
    zero_grad impede acumular derivadas de passos anteriores; backward calcula
    a regra da cadeia; step aplica a regra do otimizador aos parâmetros registrados.
    """
    from ..from_scratch.logistic_regression import validate_targets

    validate_targets(y, len(X))
    inputs = torch.tensor(X, dtype=torch.float64, device="cpu")
    targets = torch.tensor(y, dtype=torch.float64, device="cpu").reshape(-1, 1)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores = model(inputs)
    loss = nn.BCEWithLogitsLoss()(scores, targets)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        return float(nn.BCEWithLogitsLoss()(model(inputs), targets).item())


def save_state_dict(model: EquipmentMLP, path: "Path") -> None:
    """Salva somente state_dict CPU, sem serializar a classe Python inteira.

    Este arquivo não contém escalas nem arquitetura: precisa do código da classe
    e do JSON de pré-processamento da prática. Não é checkpoint de treinamento.
    Cria pais e substitui destino. Use somente artefatos produzidos por você.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state_dict(path: "Path") -> EquipmentMLP:
    """Reconstrói arquitetura fixa e lê pesos com weights_only=True e CPU.

    Rejeita arquivos maiores que 1 MB, chaves/formatos incompatíveis e valores
    não finitos. weights_only reduz superfícies de execução, mas não autentica
    arquivos nem elimina ataques de recursos; não amplie allowlist para “resolver”
    falha ao carregar um arquivo desconhecido. Retorna modelo em eval.
    """
    if path.stat().st_size > 1_000_000:
        raise ValueError("O state_dict didático excede 1 MB.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = EquipmentMLP()
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Chaves do state_dict incompatíveis.")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != torch.float64 or not bool(torch.isfinite(value).all())):
            raise ValueError("Pesos têm forma, dtype ou finitude incompatíveis.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
