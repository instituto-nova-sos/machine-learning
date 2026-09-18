"""Regressão logística binária em Python puro, sem NumPy ou Scikit-learn.

Uma linha representa um exemplo; cada coluna, um atributo. Primeiro calculamos
``z = soma(w_j * x_j) + b``; depois ``p = sigmoid(z)``. A perda é a entropia cruzada
binária média, sem regularização. As entradas devem ser padronizadas fora do modelo,
usando apenas o treino. Os pesos passam a expressar mudanças por unidade padronizada.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .activations import probability_to_class, sigmoid


def validate_features(features: Sequence[Sequence[float]], width: int) -> None:
    """Exige matriz não vazia, retangular, finita e com ``width`` atributos por linha.

    Levanta ValueError para formas ou valores inválidos. Não altera nem padroniza dados.
    A ordem das colunas é um contrato do chamador: trocar temperatura e vibração mantém
    a forma, mas muda o significado e produz previsões incorretas.
    """
    if width < 1 or len(features) == 0:
        raise ValueError("Forneça exemplos e pelo menos um atributo.")
    if any(len(row) != width for row in features):
        raise ValueError("Cada exemplo deve ter o mesmo número de atributos do modelo.")
    if any(not math.isfinite(value) for row in features for value in row):
        raise ValueError("Os atributos devem ser finitos.")


def validate_targets(targets: Sequence[int], size: int) -> None:
    """Exige um rótulo 0 ou 1 por exemplo; incompatibilidades levantam ValueError.

    Um lote com uma só classe é válido para calcular a perda e seus gradientes.
    Já o treinamento didático completo exige as duas classes, conforme ``fit``.
    """
    if len(targets) != size or size == 0:
        raise ValueError("Entradas e alvos devem ter o mesmo tamanho não nulo.")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("Os alvos devem ser classes 0 ou 1.")


def binary_cross_entropy_from_logits(scores: Sequence[float], targets: Sequence[int]) -> float:
    """Retorna a perda média, sem unidade, calculada diretamente dos escores.

    A fórmula conceitual é ``-[y*log(p) + (1-y)*log(1-p)]``. Aqui usamos a forma
    equivalente ``max(z, 0) - y*z + log1p(exp(-abs(z)))``. Ela evita log(0), mesmo
    quando a sigmoid arredonda para 0 ou 1, e evita overflow na exponencial.
    ``log1p(t)`` calcula log(1+t) com precisão quando t é pequeno.

    Escores e rótulos têm forma ``(n_exemplos,)`` e a mesma ordem. Levanta ValueError
    para dados vazios, comprimentos diferentes, rótulos não binários ou escores não
    finitos. Não recorta probabilidades: uma previsão confiante e errada continua
    recebendo penalidade grande (escore 1000 com alvo 0 tem perda aproximadamente 1000).
    """
    validate_targets(targets, len(scores))
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Os escores devem ser finitos; verifique escala e taxa de aprendizado.")
    # Dividir cada parcela antes da soma também reduz o risco de overflow na redução.
    return sum(
        (max(z, 0.0) - y * z + math.log1p(math.exp(-abs(z)))) / len(scores)
        for z, y in zip(scores, targets, strict=True)
    )


@dataclass
class LogisticRegressionBinary:
    """Classificador linear binário, com pesos explícitos e viés inicial zero.

    ``weights`` contém um peso por coluna; ``bias`` desloca o escore. Exemplo:
    ``LogisticRegressionBinary([0.0, 0.0])`` começa com probabilidade 0,5 para qualquer
    linha com dois atributos. A instância copia a lista inicial para não modificar
    a lista do chamador durante o treino. O ajuste modifica os parâmetros da instância.
    Não há persistência, seleção de limiar ou validação cruzada nesta classe.
    """

    weights: list[float]
    bias: float = 0.0

    def __post_init__(self) -> None:
        """Copia pesos e rejeita parâmetros vazios ou não finitos com ValueError."""
        self.weights = list(self.weights)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Verifica também parâmetros alterados após a construção ou durante o treino."""
        if not self.weights or any(not math.isfinite(w) for w in self.weights):
            raise ValueError("Forneça pelo menos um peso, e todos devem ser finitos.")
        if not math.isfinite(self.bias):
            raise ValueError("O viés deve ser finito.")

    def decision_function(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Calcula um escore ``z`` por linha, sem aplicar sigmoid nem limiar.

        Entradas têm forma (n, d), com d igual ao número de pesos. A saída preserva
        as n linhas. Formas incompatíveis ou valores não finitos geram ValueError.
        """
        self._validate_parameters()
        validate_features(features, len(self.weights))
        scores = [
            sum(w * x for w, x in zip(self.weights, row, strict=True)) + self.bias
            for row in features
        ]
        if any(not math.isfinite(z) for z in scores):
            raise ValueError("Os escores devem ser finitos; padronize as entradas.")
        return scores

    def predict_proba(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Retorna apenas P(classe=1) por linha, não uma matriz de duas classes.

        São estimativas condicionadas aos atributos, modelo e dados; a sigmoid garante
        o intervalo, mas não garante calibração nem validade em equipamentos reais.
        """
        return [sigmoid(z) for z in self.decision_function(features)]

    def predict(self, features: Sequence[Sequence[float]], *, threshold: float = 0.5) -> list[int]:
        """Aplica o limiar inclusivo às probabilidades; não modifica nem treina o modelo.

        ``threshold`` deve pertencer a [0, 1] e ser finito, ou ocorre ValueError.
        O padrão 0,5 é uma convenção didática, não uma decisão operacional validada.
        """
        return [probability_to_class(p, threshold=threshold) for p in self.predict_proba(features)]

    def loss(self, features: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
        """Mede entropia cruzada média com o estado atual, sem atualizar parâmetros."""
        return binary_cross_entropy_from_logits(self.decision_function(features), targets)

    def gradients(
        self, features: Sequence[Sequence[float]], targets: Sequence[int]
    ) -> tuple[list[float], float]:
        """Calcula ``dw_j = média((p-y)*x_j)`` e ``db = média(p-y)`` no lote.

        Ao derivar a entropia cruzada composta com sigmoid, fatores se cancelam e
        deixam o resíduo ``p-y``. Não é o gradiente do MSE da trilha anterior.
        Todos os gradientes usam os mesmos pesos, antes de qualquer atualização.
        Retorna d derivadas dos pesos e uma do viés; não altera o estado.
        """
        probabilities = self.predict_proba(features)
        validate_targets(targets, len(features))
        errors = [p - y for p, y in zip(probabilities, targets, strict=True)]
        count = len(features)
        weight_gradients = [
            sum(error * row[j] / count for error, row in zip(errors, features, strict=True))
            for j in range(len(self.weights))
        ]
        return weight_gradients, sum(error / count for error in errors)

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[int],
        *,
        learning_rate: float = 0.1,
        epochs: int = 1000,
    ) -> list[float]:
        """Executa gradiente descendente em lote e retorna perda após cada atualização.

        Requer X finito (n, d), y binário (n,) com ambas as classes, taxa finita
        positiva e épocas inteiras positivas. Violações levantam ValueError.
        Uma nova chamada continua dos pesos atuais: não reinicializa a instância.
        Os padrões são escolhas didáticas para entradas padronizadas, não garantias
        de convergência. Dados perfeitamente separáveis podem levar pesos a crescer
        indefinidamente sem regularização; perda menor não garante generalização.
        """
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("A taxa de aprendizado deve ser finita e positiva.")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("O número de épocas deve ser inteiro e positivo.")
        validate_features(features, len(self.weights))
        validate_targets(targets, len(features))
        if set(targets) != {0, 1}:
            raise ValueError("O treino deve conter exemplos das duas classes.")
        history = []
        for _ in range(epochs):
            dw, db = self.gradients(features, targets)
            # Atualização simultânea: nenhum gradiente usa um peso já atualizado.
            self.weights = [
                w - learning_rate * gradient for w, gradient in zip(self.weights, dw, strict=True)
            ]
            self.bias -= learning_rate * db
            history.append(self.loss(features, targets))
        return history
