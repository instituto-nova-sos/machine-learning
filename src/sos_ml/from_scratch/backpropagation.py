"""Regra da cadeia escalar, retropropagação matricial e diferenças finitas.

Backpropagation calcula gradientes. Gradiente descendente usa esses gradientes
para atualizar parâmetros: são etapas diferentes, aqui em funções diferentes.
"""

import math

import numpy as np
from numpy.typing import ArrayLike

from .dense_layer import FloatArray, matrix
from .logistic_regression import validate_targets
from .neural_network import BinaryMLP


def scalar_example(x: float = 2, w: float = 0.5, b: float = 0.1,
                   target: float = 1) -> dict[str, float]:
    """Expõe z=wx+b, a=z², J=(a-y)²/2 e cada derivada local em Python puro.

    x,y,w,b são escalares didáticos sem unidade; não é BCE nem a MLP ReLU.
    Retorna intermediários e dJ/dw, dJ/db; não altera nada. Rejeita não finitos.
    A divisão por 2 cancela o fator 2 da derivada da perda quadrática.
    """
    if any(not math.isfinite(v) for v in (x, w, b, target)):
        raise ValueError("Os valores escalares devem ser finitos.")
    z = w * x + b
    a = z**2
    dloss_da = a - target
    da_dz = 2 * z
    dloss_dz = dloss_da * da_dz
    return {"z": z, "a": a, "perda": (a - target)**2 / 2,
            "dJ_da": dloss_da, "da_dz": da_dz, "dJ_dz": dloss_dz,
            "dJ_dw": dloss_dz * x, "dJ_db": dloss_dz}


def gradients(model: BinaryMLP, inputs: ArrayLike, targets: list[int]) -> list[FloatArray]:
    """Calcula [dW₁, db₁, dW₂, db₂] da BCE média sem alterar os parâmetros.

    X(n,d); H(n,h); logits(n,1); y(n,1). O resíduo D₂=(sigmoid(z)-y)/n
    já incorpora a derivada da perda e a média. D₁=(D₂@W₂.T)*(Z₁>0)
    aplica a derivada local da ReLU. Cada transposta acumula contribuições dos
    exemplos, enquanto a soma no eixo 0 acumula derivadas dos vieses.
    Em Z₁=0 escolhemos derivada zero; gradient check deve evitar essa dobra.
    """
    model.__post_init__()
    X = matrix(inputs, model.hidden.weights.shape[0])
    validate_targets(targets, len(X))
    # Forward completo valida também finitude dos parâmetros e das operações.
    scores = model.logits(X)
    hidden_scores = X @ model.hidden.weights + model.hidden.bias
    hidden = np.maximum(hidden_scores, 0)
    probabilities = np.exp(-np.logaddexp(0, -scores))
    labels: FloatArray = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    delta_output = (probabilities - labels) / len(X)
    dw_output = hidden.T @ delta_output
    db_output = delta_output.sum(axis=0)
    delta_hidden = (delta_output @ model.output.weights.T) * (hidden_scores > 0)
    return [X.T @ delta_hidden, delta_hidden.sum(axis=0), dw_output, db_output]


def parameters(model: BinaryMLP) -> list[FloatArray]:
    """Retorna referências na ordem [W₁,b₁,W₂,b₂], não cópias.

    Essa ordem coincide com gradients. Modificar os arrays altera o modelo;
    centralizar a ordem evita aplicar uma derivada ao parâmetro errado.
    """
    return [model.hidden.weights, model.hidden.bias, model.output.weights, model.output.bias]


def gradient_check(model: BinaryMLP, inputs: ArrayLike, targets: list[int],
                   epsilon: float = 1e-6) -> float:
    """Retorna maior erro absoluto entre derivadas analíticas e diferenças centrais.

    Perturba cada escalar por ±epsilon, calcula a perda e restaura parâmetros
    mesmo em exceções. Custa duas avaliações por escalar: só use em redes pequenas.
    Float64 e epsilon 1e-6 equilibram truncamento/cancelamento neste exemplo;
    não existe tolerância universal, sobretudo perto das dobras da ReLU.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Epsilon deve ser finito e positivo.")
    analytic = gradients(model, inputs, targets)
    maximum = 0.0
    for parameter, derivative in zip(parameters(model), analytic, strict=True):
        for index in np.ndindex(parameter.shape):
            original = float(parameter[index])
            try:
                parameter[index] = original + epsilon
                plus = model.loss(inputs, targets)
                parameter[index] = original - epsilon
                minus = model.loss(inputs, targets)
            finally:
                parameter[index] = original
            maximum = max(maximum, abs(float(derivative[index]) - (plus-minus)/(2*epsilon)))
    return maximum


def fit(model: BinaryMLP, inputs: ArrayLike, targets: list[int], *,
        epochs: int = 500, learning_rate: float = 0.1) -> list[float]:
    """Gradiente descendente em lote; retorna BCE depois de cada atualização.

    Requer épocas inteiras positivas, taxa finita positiva e X/y válidos.
    Não escolhe hiperparâmetros, não usa teste e não promete convergência.
    A instância é modificada; nova chamada continua dos parâmetros atuais.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Épocas devem ser inteiras e positivas.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Taxa deve ser finita e positiva.")
    history = []
    for _ in range(epochs):
        derivatives = gradients(model, inputs, targets)
        for parameter, derivative in zip(parameters(model), derivatives, strict=True):
            parameter -= learning_rate * derivative
        history.append(model.loss(inputs, targets))
    return history
