"""Reprodutibilidade, prevenção de vazamento e execução da prática de classificação."""

import random
import subprocess
import sys
from pathlib import Path

import pytest

from sos_ml.equipment_data import generate_equipment_data, prepare_equipment_data


def test_generator_is_reproducible_and_local() -> None:
    """A semente reproduz os dados sem consumir o gerador global da aplicação."""
    state = random.getstate()
    X, y = generate_equipment_data()
    assert random.getstate() == state
    assert (X, y) == generate_equipment_data()
    assert (X, y) != generate_equipment_data(seed=43)
    assert len(X) == len(y) == 400
    assert set(y) == {0, 1}
    assert all(40 <= row[0] <= 100 and 0.5 <= row[1] <= 8 for row in X)


def test_scalers_never_see_test_data() -> None:
    """Alterar só atributos e rótulos do teste não pode afetar escala nem treino."""
    X, y = generate_equipment_data()
    split = prepare_equipment_data(X, y)
    assert not set(split.train_indices) & set(split.test_indices)
    assert sorted(split.train_indices + split.test_indices) == list(range(400))
    assert len(split.X_train) == 320
    assert len(split.X_test) == 80
    changed_X = [row[:] for row in X]
    changed_y = y[:]
    for index in split.test_indices:
        changed_X[index] = [10000.0, 10000.0]
        changed_y[index] = 1 - y[index]
    changed = prepare_equipment_data(changed_X, changed_y)
    assert changed.scalers == split.scalers
    assert changed.X_train == split.X_train
    assert changed.y_train == split.y_train
    assert changed.X_test != split.X_test
    for column in range(2):
        values = [row[column] for row in split.X_train]
        assert sum(values) / len(values) == pytest.approx(0, abs=1e-12)
        assert sum(value**2 for value in values) / len(values) == pytest.approx(1)
    assert split.transform([X[split.test_indices[0]]])[0] == split.X_test[0]


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_invalid_dataset_size(size: int) -> None:
    """Evita datasets vazios e contagens que não representam exemplos inteiros."""
    with pytest.raises(ValueError):
        generate_equipment_data(size=size)


def test_classification_cli_with_plot(tmp_path: Path) -> None:
    """Exercita o comando público em processo separado e confere o artefato PNG."""
    plot = tmp_path / "classificacao.png"
    result = subprocess.run(
        [sys.executable, "-m", "sos_ml.classify", "--plot", str(plot)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Treino: 320; teste preservado: 80" in result.stdout
    assert "Python puro:" in result.stdout
    assert "NumPy:" in result.stdout
    assert "Scikit-learn:" in result.stdout
    assert "Baseline" in result.stdout
    assert "Nova medida:" in result.stdout
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
