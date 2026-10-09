"""PyTorch é comparado às mesmas operações/derivadas manuais, sem rede externa."""

import numpy as np
import pytest

# A suíte base funciona sem o extra torch; validate-torch exige sua instalação.
torch = pytest.importorskip("torch")
from sos_ml.from_scratch.backpropagation import gradients  # noqa: E402
from sos_ml.from_scratch.neural_network import BinaryMLP  # noqa: E402
from sos_ml.torch_models.equipment import EquipmentMLP, train_step  # noqa: E402


def test_forward_loss_and_every_gradient() -> None:
    """Mesmo estado, lote e BCE devem produzir forward e gradientes equivalentes."""
    manual = BinaryMLP.initialize()
    model = EquipmentMLP()
    X, y = [[-0.7, 1.2], [0.5, -0.3], [1.4, 0.8]], [0, 1, 0]
    inputs = torch.tensor(X, dtype=torch.float64)
    targets = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
    scores = model(inputs)
    np.testing.assert_allclose(scores.detach().numpy(), manual.logits(X), atol=1e-12)
    loss = torch.nn.BCEWithLogitsLoss()(scores, targets)
    assert loss.item() == pytest.approx(manual.loss(X, y), abs=1e-12)
    loss.backward()
    actual = [model.hidden.weight.grad.T, model.hidden.bias.grad,
              model.output.weight.grad.T, model.output.bias.grad]
    for a, b in zip(actual, gradients(manual, X, y), strict=True):
        np.testing.assert_allclose(a.numpy(), b, atol=1e-12)


def test_training_and_modes() -> None:
    """Treino reduz BCE, saída sem grafo e modo eval são contratos separados."""
    torch.set_num_threads(1)
    model = EquipmentMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = [[-1., -1.], [1., 1.]], [0, 1]
    before = model.to_numpy().loss(X, y)
    for _ in range(100):
        final = train_step(model, X, y, optimizer)
    assert final < before / 2
    assert len(model.predict_proba(X)) == 2
    assert not model.training
    with torch.no_grad():
        assert not model(torch.tensor(X, dtype=torch.float64)).requires_grad


def test_state_dict_reload(tmp_path) -> None:
    """Persistir parâmetros e reconstruir classe preserva o forward."""
    from sos_ml.torch_models.equipment import load_state_dict, save_state_dict

    model = EquipmentMLP()
    path = tmp_path / "pesos.pt"
    save_state_dict(model, path)
    loaded = load_state_dict(path)
    assert loaded.predict_proba([[1, 2]]) == model.predict_proba([[1, 2]])
    assert not loaded.training
    state = model.state_dict()
    state["hidden.weight"] = torch.ones((1, 1), dtype=torch.float64)
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_state_dict(path)


def test_training_backends_equivalent() -> None:
    """Mesmo início e atualizações: duas implementações devem concordar."""
    from sos_ml.neural_training import train_equipment

    manual = train_equipment("numpy", epochs=30)
    automated = train_equipment("torch", epochs=30)
    assert automated.training_loss == pytest.approx(manual.training_loss, abs=1e-12)
    assert automated.validation_loss == pytest.approx(manual.validation_loss, abs=1e-12)
    assert automated.artifact.predict_proba([[80, 5]]) == pytest.approx(
        manual.artifact.predict_proba([[80, 5]]), abs=1e-12
    )


@pytest.mark.parametrize("fail", [False, True])
def test_training_preserves_host_threads(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    """Treino reutilizável preserva threads do hospedeiro, inclusive quando falha."""
    from sos_ml.neural_training import train_equipment

    original = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        if fail:
            def broken_step(*args, **kwargs):
                raise RuntimeError("Falha de treino simulada.")

            monkeypatch.setattr("sos_ml.torch_models.equipment.train_step", broken_step)
            with pytest.raises(RuntimeError, match="simulada"):
                train_equipment("torch", epochs=1)
        else:
            train_equipment("torch", epochs=1)
        assert torch.get_num_threads() == 2
    finally:
        # O próprio teste também deve devolver a configuração original ao processo.
        torch.set_num_threads(original)
