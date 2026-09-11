"""Geração e carregamento dos dados sintéticos do projeto de regressão.

O módulo concentra o contrato tabular usado pelo restante da aplicação: cada linha representa
um imóvel fictício, ``area_m2`` é o único atributo de entrada e ``preco_brl`` é o alvo contínuo.
Os números existem apenas para tornar o fluxo de Machine Learning reproduzível. Eles não foram
coletados do mercado e, portanto, não fundamentam avaliações ou decisões financeiras reais.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_housing_data(size: int = 120, seed: int = 42) -> pd.DataFrame:
    """Gera exemplos fictícios de área e preço para a regressão didática.

    Cada alvo segue ``preco = 80_000 + 3_200 * area + ruido``. A relação linear conhecida
    permite verificar se o algoritmo recupera aproximadamente a inclinação e o intercepto,
    enquanto o ruído gaussiano impede que o exercício seja uma linha perfeitamente artificial.

    Args:
        size: Número de linhas do conjunto. O padrão é 120; são exigidas ao menos 10 para que
            os exemplos posteriores tenham partições de treino e teste minimamente úteis.
        seed: Estado inicial do gerador pseudoaleatório. A mesma semente produz os mesmos dados,
            o que torna aulas, testes e resultados reproduzíveis.

    Returns:
        DataFrame com ``size`` linhas e duas colunas: ``area_m2``, em metros quadrados, e
        ``preco_brl``, em reais fictícios. Ambos os valores são arredondados para duas casas.

    Raises:
        ValueError: Se ``size`` for menor que 10.

    Warning:
        A fórmula e seus parâmetros foram escolhidos para ensino. O resultado não representa a
        formação de preços de nenhum mercado imobiliário real.
    """
    if size < 10:
        # Uma amostra muito curta tornaria a separação posterior pouco instrutiva, mesmo que a
        # geração matemática ainda fosse possível.
        raise ValueError("Gere pelo menos 10 exemplos para permitir treino e avaliação.")

    # ``default_rng`` cria um gerador local: reproduzimos a sequência sem modificar o estado
    # aleatório global de outras bibliotecas ou partes do programa.
    rng = np.random.default_rng(seed)

    # A distribuição uniforme dá a todas as faixas do intervalo didático [30, 180) m² a mesma
    # chance de aparecer, ajudando a reta a observar entradas pequenas e grandes.
    area = rng.uniform(30.0, 180.0, size=size)

    # O ruído tem média zero e desvio-padrão de R$ 18 mil. Ele representa, de forma abstrata,
    # fatores omitidos do modelo de um único atributo; não é uma estimativa obtida do mundo real.
    noise = rng.normal(0.0, 18_000.0, size=size)

    # 80 mil é o intercepto e 3.200 reais/m² é a inclinação escolhida para a relação sintética.
    price = 80_000.0 + 3_200.0 * area + noise

    # Duas casas decimais imitam a representação usual das unidades e estabilizam o CSV gerado.
    return pd.DataFrame({"area_m2": area.round(2), "preco_brl": price.round(2)})


def load_housing_data(path: Path) -> pd.DataFrame:
    """Lê um CSV e valida o contrato mínimo do projeto de imóveis.

    Args:
        path: Caminho do arquivo CSV. Colunas adicionais são mantidas, mas ``area_m2`` e
            ``preco_brl`` precisam existir e não podem conter valores ausentes.

    Returns:
        DataFrame na ordem original das linhas e colunas.

    Raises:
        FileNotFoundError: Se ``path`` não existir.
        ValueError: Se faltar uma coluna obrigatória ou houver ausência nessas colunas.

    Note:
        Esta função valida somente o contrato necessário nesta etapa. Tipos numéricos, limites
        físicos e duplicatas seriam validações importantes em um sistema de produção.
    """
    frame = pd.read_csv(path)

    # Um conjunto elimina dependência da ordem das colunas durante a verificação de presença.
    required = {"area_m2", "preco_brl"}
    if not required.issubset(frame.columns):
        raise ValueError(f"O CSV deve conter as colunas: {sorted(required)}.")

    # O modelo e a perda não definem uma política de imputação; rejeitar ausências evita que NaN
    # se propague silenciosamente por previsões, gradientes e métricas.
    if frame[list(required)].isna().any().any():
        raise ValueError("O dataset contém valores ausentes nas colunas obrigatórias.")
    return frame
