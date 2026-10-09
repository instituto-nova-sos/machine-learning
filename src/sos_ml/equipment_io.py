"""Artefato JSON da MLP: contrato, pesos e pré-processamento juntos e inspecionáveis.

JSON não executa código na leitura, mas não autentica origem. Ainda validamos
versão, formas, números finitos e limite de tamanho; não carregue arquivos de
origem desconhecida como se fossem modelos aprovados para um domínio real.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .equipment_data import transform_equipment_features
from .from_scratch.backpropagation import parameters
from .from_scratch.dense_layer import FloatArray
from .from_scratch.neural_network import BinaryMLP
from .from_scratch.scaling import StandardScaler1D

FEATURES = ["temperatura_c", "vibracao_mm_s"]


@dataclass
class EquipmentArtifact:
    """Modelo 2→4→1 e dois scalers na ordem temperatura/vibração.

    ``predict_proba`` recebe unidades físicas (n,2), aplica transformação fixa e
    retorna probabilidades; não treina. A camada de política decide usos/faixas.
    """

    model: BinaryMLP
    scalers: list[StandardScaler1D]

    def predict_proba(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Inferência local sem rótulos ou reajuste de estatísticas."""
        return self.model.predict_proba(transform_equipment_features(features, self.scalers))


def validate_scalers(scalers: list[StandardScaler1D]) -> None:
    """Exige duas médias finitas e dois desvios finitos positivos, ou ValueError."""
    if len(scalers) != 2 or any(
        not math.isfinite(s.mean) or not math.isfinite(s.scale) or s.scale <= 0 for s in scalers
    ):
        raise ValueError("Artefato exige dois scalers finitos com escala positiva.")


def save_equipment_model(artifact: EquipmentArtifact, path: Path) -> None:
    """Salva versão 1, arquitetura fixa, ordem de colunas, scalers e parâmetros.

    Cria pais/substitui destino. Não guarda otimizador: é artefato de inferência,
    não checkpoint para continuar treino. Usa allow_nan=False para não gravar
    extensões JSON não padronizadas; erros de estado ocorrem antes da escrita.
    """
    validate_scalers(artifact.scalers)
    artifact.model.logits([[0, 0]])
    payload = {
        "versao": 1, "tipo": "mlp_equipamento_sintetico", "arquitetura": [2, 4, 1],
        "colunas": FEATURES, "sintetico": True,
        "scalers": [{"media": s.mean, "escala": s.scale} for s in artifact.scalers],
        "parametros": [p.tolist() for p in parameters(artifact.model)],
    }
    # A versão 1 tem arquitetura fixa. Não transforme qualquer MLP em arquivo de
    # aparência válida: o contrato é também verificado ao escrever.
    expected = [(2, 4), (4,), (4, 1), (1,)]
    if [p.shape for p in parameters(artifact.model)] != expected:
        raise ValueError("Este esquema aceita somente arquitetura 2→4→1.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def load_equipment_model(path: Path) -> EquipmentArtifact:
    """Valida e reconstrói parâmetros/preprocessamento sem desserializar classes.

    Aceita somente nosso esquema/arquitetura, com no máximo 1 MB. FileNotFoundError
    e erros de JSON são propagados; estrutura/números incompatíveis geram ValueError.
    O limite é proteção básica de recursos, não sandbox universal para arquivos hostis.
    """
    if path.stat().st_size > 1_000_000:
        raise ValueError("Artefato didático excede o limite de 1 MB.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or (
        payload.get("versao") != 1 or payload.get("arquitetura") != [2, 4, 1]
        or payload.get("tipo") != "mlp_equipamento_sintetico"
        or payload.get("colunas") != FEATURES or payload.get("sintetico") is not True
    ):
        raise ValueError("Esquema, arquitetura ou ordem de colunas incompatíveis.")
    try:
        scalers = [StandardScaler1D(float(s["media"]), float(s["escala"]))
                   for s in payload["scalers"]]
        validate_scalers(scalers)
        values = payload["parametros"]
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("Forneça quatro arrays de parâmetros.")
        model = BinaryMLP.initialize()
        for target, source in zip(parameters(model), values, strict=True):
            array: FloatArray = np.asarray(source, dtype=np.float64)
            if array.shape != target.shape or not np.all(np.isfinite(array)):
                raise ValueError("Forma ou finitude de parâmetro incompatível.")
            target[:] = array
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Conteúdo do artefato inválido.") from error
    return EquipmentArtifact(model, scalers)
