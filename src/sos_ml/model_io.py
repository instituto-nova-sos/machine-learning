"""Persistência do modelo linear em um artefato JSON simples e legível.

O formato guarda apenas o tipo e os parâmetros necessários para reproduzir a inferência. JSON foi
escolhido por ser inspecionável pelos estudantes e por não executar código durante a leitura, ao
contrário de certos formatos de serialização Python.
"""

import json
from pathlib import Path

from .from_scratch.linear_regression import LinearRegression1D


def save_model(model: LinearRegression1D, path: Path) -> None:
    """Salva peso, viés e identificador de tipo em UTF-8.

    Args:
        model: Modelo já expresso nas unidades originais da entrada e do alvo.
        path: Destino do JSON. Diretórios pais ausentes são criados automaticamente.

    Side Effects:
        Cria diretórios e grava ou substitui o arquivo indicado.

    Note:
        O artefato é intencionalmente mínimo e ainda não possui versão de esquema, metadados de
        treino ou assinatura de integridade, recursos recomendáveis em sistemas de produção.
    """
    # ``parents=True`` cria toda a cadeia necessária; ``exist_ok=True`` mantém diretórios atuais.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"tipo": "regressao_linear_1d", "peso": model.weight, "vies": model.bias},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_model(path: Path) -> LinearRegression1D:
    """Reconstrói um modelo linear a partir do artefato JSON.

    Args:
        path: Arquivo produzido por :func:`save_model`.

    Returns:
        ``LinearRegression1D`` pronto para inferência, com peso e viés convertidos para ``float``.

    Raises:
        FileNotFoundError: Se o artefato não existir.
        json.JSONDecodeError: Se o conteúdo não for JSON válido.
        ValueError: Se o identificador declarar outro tipo de modelo.
        KeyError: Se faltarem ``peso`` ou ``vies``.

    Warning:
        Validar o campo ``tipo`` reduz erros acidentais, mas não autentica a origem do arquivo.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("tipo") != "regressao_linear_1d":
        raise ValueError("Tipo de modelo incompatível.")
    return LinearRegression1D(weight=float(payload["peso"]), bias=float(payload["vies"]))
