"""Comparação profissional com o mesmo objetivo não regularizado das versões manuais."""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from sos_ml.from_scratch.logistic_regression import validate_features, validate_targets


def fit_logistic_regression(
    features: NDArray[np.float64], targets: NDArray[np.int64]
) -> LogisticRegression:
    """Ajusta sklearn em X (n, d) e y (n,), já separados e escalados pelo chamador.

    Requer atributos finitos e ambas as classes 0/1; contratos inválidos levantam
    ValueError. Retorna estimador ajustado com coef_ (1, d) e intercept_ (1,).
    Seu predict_proba retorna (n, 2), na ordem de classes_, diferente da nossa API
    que devolve somente P(y=1). Não há partição nem reajuste de scaler aqui.

    C=infinito zera a força da penalização L2 padrão, permitindo comparar com a
    perda manual sem regularização. Isso não recomenda remover regularização em
    aplicações. L-BFGS é outro otimizador: max_iter=2000 não significa as mesmas
    2000 épocas manuais. tol=1e-9 reduz erro numérico na comparação didática, sem
    garantir igualdade bit a bit. Nenhuma escolha é ajustada olhando o teste.
    """
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("Use matriz de atributos 2D e vetor de alvos 1D.")
    validate_features(features.tolist(), features.shape[1])
    validate_targets(targets.tolist(), len(features))
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("O treino deve conter exemplos das duas classes.")
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, tol=1e-9)
    model.fit(features, targets)
    return model
