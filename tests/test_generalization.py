"""As partições e o experimento devem demonstrar o fenômeno anunciado."""

from sos_ml.from_scratch.data_split import train_validation_test_indices
from sos_ml.generalization import experiment


def test_triple_split() -> None:
    """Todos os exemplos aparecem uma vez e nenhum conjunto invade outro."""
    train, validation, test = train_validation_test_indices(400)
    assert [len(train), len(validation), len(test)] == [240, 80, 80]
    assert len(set(train + validation + test)) == 400
    assert (train, validation, test) == train_validation_test_indices(400)


def test_overfitting_is_visible() -> None:
    """Mais capacidade reduz erro de treino e piora a validação neste experimento."""
    _, train, validation, test = experiment()
    assert train[-1] < train[1]
    assert validation[-1] > min(validation)
    assert 0 <= test <= 1
