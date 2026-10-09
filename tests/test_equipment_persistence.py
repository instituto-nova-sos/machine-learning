"""Persistência inclui escala e contrato; alterar só teste não muda o treino."""

import json
from pathlib import Path

import pytest

from sos_ml.equipment_data import generate_equipment_data, prepare_development_data
from sos_ml.equipment_io import load_equipment_model, save_equipment_model
from sos_ml.neural_training import train_equipment


def test_development_has_no_leakage() -> None:
    """Alterações em validação/teste não afetam médias, desvios ou dados de treino."""
    X, y = generate_equipment_data()
    split = prepare_development_data(X, y)
    for i in split.validation_indices + split.test_indices:
        X[i] = [10000, 10000]
        y[i] = 1 - y[i]
    changed = prepare_development_data(X, y)
    assert changed.scalers == split.scalers
    assert changed.X_train == split.X_train
    assert changed.y_train == split.y_train


def test_reload_and_schema(tmp_path: Path) -> None:
    """JSON recria a mesma previsão em unidades físicas e rejeita ordem errada."""
    result = train_equipment(epochs=50)
    destination = tmp_path / "mlp.json"
    save_equipment_model(result.artifact, destination)
    loaded = load_equipment_model(destination)
    assert loaded.predict_proba([[80, 5]]) == result.artifact.predict_proba([[80, 5]])
    payload = json.loads(destination.read_text())
    payload["colunas"].reverse()
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="colunas"):
        load_equipment_model(destination)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_artifact_parameters(tmp_path: Path, invalid: float) -> None:
    """Números não finitos não devem virar probabilidades de aparência válida."""
    result = train_equipment(epochs=1)
    destination = tmp_path / "mlp.json"
    save_equipment_model(result.artifact, destination)
    payload = json.loads(destination.read_text())
    payload["parametros"][0][0][0] = invalid
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_equipment_model(destination)


def test_restore_best_validation_state() -> None:
    """O artefato é o melhor estado, não o último ponto da curva de treino."""
    result = train_equipment(epochs=300, patience=20)
    X, y = generate_equipment_data()
    split = prepare_development_data(X, y)
    assert result.best_epoch < len(result.validation_loss) - 1
    assert result.artifact.model.loss(split.X_validation, split.y_validation) == pytest.approx(
        min(result.validation_loss), abs=1e-8
    )


def test_test_labels_do_not_select_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alterar só y do teste pode mudar métricas finais, mas nunca escolhas/pesos."""
    baseline = train_equipment(epochs=30)
    X, y = generate_equipment_data()
    split = prepare_development_data(X, y)
    for i in split.test_indices:
        y[i] = 1 - y[i]
    monkeypatch.setattr("sos_ml.neural_training.generate_equipment_data", lambda: (X, y))
    altered = train_equipment(epochs=30)
    assert altered.best_epoch == baseline.best_epoch
    assert altered.training_loss == baseline.training_loss
    assert altered.validation_loss == baseline.validation_loss
    assert altered.artifact.predict_proba([[80, 5]]) == baseline.artifact.predict_proba([[80, 5]])
    assert altered.test_metrics != baseline.test_metrics
