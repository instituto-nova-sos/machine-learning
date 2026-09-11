"""Teste de integração do treinamento e da inferência pelas funções das CLIs."""

from pathlib import Path

import pandas as pd
import pytest

from sos_ml.predict import main as predict_main
from sos_ml.train import main as train_main


def test_train_and_predict_cli(tmp_path: Path, capsys: object) -> None:
    """Treina em CSV temporário, comprova o artefato e realiza uma inferência completa.

    ``tmp_path`` isola os arquivos do teste; ``capsys`` faz o pytest capturar as mensagens das CLIs
    para que elas não poluam a saída da suíte, mesmo que este teste não inspecione seu conteúdo.
    """
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.json"
    # A relação exata ``y = 2x + 1`` permite um treino curto e previsível sem o dataset completo.
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


@pytest.mark.parametrize("invalid_area", ["0", "-1", "nan", "inf", "-inf"])
def test_predict_cli_rejects_non_positive_or_non_finite_area(
    tmp_path: Path, invalid_area: str
) -> None:
    """Rejeita medidas sem significado físico antes mesmo de tentar carregar o modelo.

    O tipo ``float`` do argparse converte representações IEEE 754 como ``nan`` e ``inf`` sem gerar
    erro. O teste documenta que ser conversível para ponto flutuante não basta: a área precisa ser
    finita e estritamente positiva. Um caminho inexistente comprova também que a validação da
    entrada ocorre antes da leitura do artefato.
    """
    missing_model = tmp_path / "modelo-inexistente.json"

    # ``ArgumentParser.error`` encerra a CLI com código 2, convenção usada para argumento inválido.
    with pytest.raises(SystemExit) as error:
        predict_main(["--model", str(missing_model), "--area", invalid_area])

    assert error.value.code == 2
