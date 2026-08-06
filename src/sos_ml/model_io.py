"""Serialização simples e legível do modelo didático."""

import json
from pathlib import Path

from .from_scratch.linear_regression import LinearRegression1D


def save_model(model: LinearRegression1D, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"tipo": "regressao_linear_1d", "peso": model.weight, "vies": model.bias},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_model(path: Path) -> LinearRegression1D:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("tipo") != "regressao_linear_1d":
        raise ValueError("Tipo de modelo incompatível.")
    return LinearRegression1D(weight=float(payload["peso"]), bias=float(payload["vies"]))
