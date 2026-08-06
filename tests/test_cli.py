from pathlib import Path

import pandas as pd

from sos_ml.predict import main as predict_main
from sos_ml.train import main as train_main


def test_train_and_predict_cli(tmp_path: Path, capsys: object) -> None:
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.json"
    pd.DataFrame({"area_m2": [1, 2, 3, 4], "preco_brl": [3, 5, 7, 9]}).to_csv(
        data_path, index=False
    )
    assert train_main(
        [
            "--data",
            str(data_path),
            "--output",
            str(model_path),
            "--epochs",
            "1000",
            "--learning-rate",
            "0.05",
        ]
    ) == 0
    assert model_path.exists()
    assert predict_main(["--model", str(model_path), "--area", "5"]) == 0
