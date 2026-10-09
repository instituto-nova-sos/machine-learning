"""Novo processo só faz inferência; logs e gráficos são contratos observáveis."""

import json
import subprocess
import sys
from pathlib import Path

from sos_ml.equipment_io import save_equipment_model
from sos_ml.neural_training import train_equipment


def test_reload_in_new_process_and_log(tmp_path: Path) -> None:
    """Verifica probabilidade idêntica, confirmação obrigatória e log estruturado."""
    trained = train_equipment(epochs=30)
    model, log = tmp_path / "modelo.json", tmp_path / "eventos.jsonl"
    save_equipment_model(trained.artifact, model)
    before = model.read_bytes()
    result = subprocess.run([
        sys.executable, "-m", "sos_ml.local_ai", "inferir", "--model", str(model),
        "--temperature", "80", "--vibration", "5", "--high-consequence", "--log", str(log),
    ], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout.splitlines()[0])
    assert report["decisao"]["probability"] == trained.artifact.predict_proba([[80, 5]])[0]
    assert report["politica"]["requires_human"]
    assert json.loads(log.read_text()) == report
    assert model.read_bytes() == before
    assert "Melhor época" not in result.stdout
