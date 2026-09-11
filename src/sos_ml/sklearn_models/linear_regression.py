"""Equivalente em Scikit-learn da regressão linear implementada manualmente."""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression


def fit_linear_regression(
    features: NDArray[np.float64], targets: NDArray[np.float64]
) -> LinearRegression:
    """Valida formas e ajusta ``sklearn.linear_model.LinearRegression``.

    Args:
        features: Matriz ``X`` de ``float64`` com forma ``(n_exemplos, n_atributos)``. Mesmo com um
            único atributo, Scikit-learn exige o eixo de colunas, por exemplo ``[[30.0], [45.0]]``.
        targets: Vetor ``y`` de ``float64`` com forma ``(n_exemplos,)``.

    Returns:
        Estimador ajustado. Seus atributos ``coef_`` e ``intercept_`` podem ser comparados ao peso e
        ao viés da implementação manual.

    Raises:
        ValueError: Se ``X`` não for bidimensional, ``y`` não for unidimensional ou as quantidades
            de exemplos divergirem.

    Note:
        Esta função não divide nem padroniza dados. Essas decisões devem ocorrer antes do ajuste e
        estatísticas de transformação precisam ser aprendidas somente no treino.
    """
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("Use features 2D, targets 1D e o mesmo número de exemplos.")
    # O estimador inclui intercepto por padrão, correspondendo ao ``bias`` do modelo manual.
    model = LinearRegression()
    model.fit(features, targets)
    return model
