"""Equivalente Scikit-learn de from_scratch/linear_regression.py."""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression


def fit_linear_regression(
    features: NDArray[np.float64], targets: NDArray[np.float64]
) -> LinearRegression:
    """Ajusta sklearn após validar a forma (n_exemplos, n_atributos)."""
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("Use features 2D, targets 1D e o mesmo número de exemplos.")
    model = LinearRegression()
    model.fit(features, targets)
    return model
