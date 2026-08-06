"""Geração e carregamento de dados sintéticos do projeto progressivo."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_housing_data(size: int = 120, seed: int = 42) -> pd.DataFrame:
    """Gera relação didática área→preço; não modela um mercado real."""
    if size < 10:
        raise ValueError("Gere pelo menos 10 exemplos para permitir treino e avaliação.")
    rng = np.random.default_rng(seed)
    area = rng.uniform(30.0, 180.0, size=size)
    noise = rng.normal(0.0, 18_000.0, size=size)
    price = 80_000.0 + 3_200.0 * area + noise
    return pd.DataFrame({"area_m2": area.round(2), "preco_brl": price.round(2)})


def load_housing_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"area_m2", "preco_brl"}
    if not required.issubset(frame.columns):
        raise ValueError(f"O CSV deve conter as colunas: {sorted(required)}.")
    if frame[list(required)].isna().any().any():
        raise ValueError("O dataset contém valores ausentes nas colunas obrigatórias.")
    return frame
