"""Dados inteiramente sintéticos para estudar classificação de falhas em CPU.

Cada linha é uma observação independente de um equipamento fictício, feita antes do
intervalo de 24 horas no qual definimos o alvo. Não há identificadores de pessoas nem
medições de máquinas reais. As regras abaixo foram inventadas para ensinar ML.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .from_scratch.activations import sigmoid
from .from_scratch.data_split import train_test_indices
from .from_scratch.logistic_regression import validate_features, validate_targets
from .from_scratch.scaling import StandardScaler1D


def generate_equipment_data(
    *, size: int = 400, seed: int = 42
) -> tuple[list[list[float]], list[int]]:
    """Gera X (n, 2) e y (n,), determinísticos para os mesmos argumentos.

    Colunas de X: temperatura entre 40 e 100 °C; vibração entre 0,5 e 8 mm/s.
    y=1 representa falha nas próximas 24 horas; y=0, ausência de falha nesse período.
    O tamanho deve ser inteiro positivo (caso contrário, ValueError). A semente 42
    permite reproduzir a aula sem alterar o gerador aleatório global do Python.

    A regra artificial é z=-1 + 1,2*(T-70)/15 + 1,4*(V-4)/2. As constantes centram
    e escalam valores apenas para definir o mundo sintético; NÃO são estatísticas
    estimadas do dataset nem substituem o scaler ajustado no treino. Sorteamos o
    rótulo por Bernoulli(p): y=1 quando um uniforme em [0,1) é menor que sigmoid(z).
    Assim, condições parecidas podem ter desfechos diferentes. O classificador não
    recebe a probabilidade geradora e não consegue prever cada sorteio individual.

    A taxa de falhas e as faixas não descrevem nenhuma indústria. Amostras pequenas
    podem conter uma só classe; o gerador não resorteia para ocultar essa limitação.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("O tamanho deve ser um inteiro positivo.")
    generator = random.Random(seed)
    features: list[list[float]] = []
    targets: list[int] = []
    for _ in range(size):
        temperature = generator.uniform(40.0, 100.0)
        vibration = generator.uniform(0.5, 8.0)
        score = -1.0 + 1.2 * (temperature - 70.0) / 15.0 + 1.4 * (vibration - 4.0) / 2.0
        target = int(generator.random() < sigmoid(score))
        features.append([temperature, vibration])
        targets.append(target)
    return features, targets


@dataclass
class EquipmentSplit:
    """Reúne partição, transformação e dados para tornar o experimento auditável.

    Índices referem-se às linhas originais; X_train e X_test têm duas colunas já
    padronizadas. Os alvos permanecem 0/1, sem padronização. Há um scaler por coluna,
    na ordem temperatura/vibração, a ser reutilizado em qualquer nova inferência.
    """

    train_indices: list[int]
    test_indices: list[int]
    scalers: list[StandardScaler1D]
    X_train: list[list[float]]
    X_test: list[list[float]]
    y_train: list[int]
    y_test: list[int]

    def transform(self, features: Sequence[Sequence[float]]) -> list[list[float]]:
        """Converte novas medidas físicas (n, 2) usando as estatísticas do treino.

        Não reajusta médias ou desvios. Entradas vazias, não finitas ou com outra
        quantidade de colunas levantam ValueError. Não verifica limites físicos de
        sensores: extrapolação e validade operacional dependem de análise do domínio.
        """
        return transform_equipment_features(features, self.scalers)


def transform_equipment_features(
    features: Sequence[Sequence[float]], scalers: Sequence[StandardScaler1D]
) -> list[list[float]]:
    """Aplica exatamente dois scalers, um por coluna, sem ajustar estatística nova.

    É compartilhada pela preparação e pela inferência. Violações do contrato de
    forma geram ValueError; os scalers devem vir de StandardScaler1D.fit no treino.
    """
    if len(scalers) != 2:
        raise ValueError("Forneça um scaler para temperatura e outro para vibração.")
    validate_features(features, 2)
    return [
        [scaler.transform([value])[0] for scaler, value in zip(scalers, row, strict=True)]
        for row in features
    ]


def prepare_equipment_data(
    features: Sequence[Sequence[float]], targets: Sequence[int], *, seed: int = 17
) -> EquipmentSplit:
    """Separa 80%/20% e ajusta os dois scalers SOMENTE nas linhas de treino.

    A semente da partição é distinta da geração para evidenciar decisões separadas.
    A divisão aleatória simples serve ao mundo artificial de observações independentes.
    Em registros repetidos de máquinas reais seria necessário considerar grupos e
    tempo. O teste pode conter uma só classe; o treino precisa conter ambas.
    Levanta ValueError para entradas inválidas, treino com uma classe ou coluna
    constante. As duas implementações manuais e o sklearn usarão esta mesma saída.
    """
    validate_features(features, 2)
    validate_targets(targets, len(features))
    train, test = train_test_indices(len(features), seed=seed)
    y_train = [targets[i] for i in train]
    if set(y_train) != {0, 1}:
        raise ValueError("A partição de treino precisa conter as duas classes.")
    scalers = [StandardScaler1D.fit([features[i][j] for i in train]) for j in range(2)]
    return EquipmentSplit(
        train_indices=train,
        test_indices=test,
        scalers=scalers,
        X_train=transform_equipment_features([features[i] for i in train], scalers),
        X_test=transform_equipment_features([features[i] for i in test], scalers),
        y_train=y_train,
        y_test=[targets[i] for i in test],
    )
