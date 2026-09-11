"""Partição determinística de índices antes de qualquer transformação dos dados.

Retornar índices, em vez de copiar os exemplos, separa a decisão de amostragem do formato do
dataset. A implementação usa apenas a biblioteca padrão para expor o mecanismo que ferramentas de
alto nível normalmente encapsulam.
"""

import random


def train_test_indices(
    size: int, *, test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Embaralha e divide ``range(size)`` em índices de treino e teste.

    Args:
        size: Quantidade total de exemplos, necessariamente maior ou igual a dois.
        test_fraction: Fração desejada para teste, estritamente entre zero e um. O arredondamento
            pode fazer a fração efetiva variar em conjuntos pequenos.
        seed: Semente do embaralhamento pseudoaleatório. Repetir os argumentos repete a partição.

    Returns:
        Par ``(train_indices, test_indices)``. As listas são disjuntas e, juntas, contêm cada
        inteiro de zero a ``size - 1`` exatamente uma vez.

    Raises:
        ValueError: Se não houver dois exemplos ou se a fração não estiver no intervalo aberto
            ``(0, 1)``.

    Note:
        A função implementa amostragem aleatória simples. Dados temporais, agrupados por pessoa ou
        equipamento e classes raras podem exigir divisão temporal, por grupo ou estratificada.
    """
    if size < 2:
        raise ValueError("A partição exige pelo menos dois exemplos.")
    if not 0 < test_fraction < 1:
        raise ValueError("A fração de teste deve estar entre 0 e 1.")
    # Um objeto Random local conserva a reprodutibilidade sem alterar o gerador global do Python.
    indices = list(range(size))
    random.Random(seed).shuffle(indices)

    # ``max`` garante ao menos um exemplo de teste. O segundo ajuste preserva pelo menos um de
    # treino quando o arredondamento, em amostras pequenas, ocuparia o conjunto inteiro.
    test_size = max(1, round(size * test_fraction))
    if test_size >= size:
        test_size = size - 1
    return indices[test_size:], indices[:test_size]
