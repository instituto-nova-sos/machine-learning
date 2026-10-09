"""Exercício Jev opcional: offline por padrão; --live solicita serviço oficial.

Sem --live, não importa SDK, não lê chave e não abre conexão. O fixture é uma
resposta inventada para testar o contrato, não resultado observado do Jev.
"""

import argparse
import json
import os
from dataclasses import asdict

from sos_ml.jev_boundary import optional_policy, parse_choice


def main() -> int:
    """Mostra fronteira mockada ou consulta SDK oficial somente sob --live explícito.

    O caminho remoto exige extra jev, TYPESAFE_API_KEY e acesso do provedor.
    Não compõe avaliação obrigatória, make validate nem rubrica. Dados enviados
    são fictícios e não representam equipamento ou pessoa real.
    """
    parser = argparse.ArgumentParser(description="Jev opcional; execução offline por padrão.")
    parser.add_argument(
        "--live", action="store_true", help="Consulta serviço oficial com sua chave."
    )
    args = parser.parse_args()
    if not args.live:
        payload = {
            "type": "choice",
            "choice": "agendar_inspecao",
            "confidence": 0.7,
            "probabilities": {
                "operacao_normal": 0.1,
                "agendar_inspecao": 0.8,
                "revisao_humana_imediata": 0.1,
            },
        }
        print("Fixture inventado: nenhuma chamada Jev foi feita.")
    else:
        if not os.environ.get("TYPESAFE_API_KEY"):
            parser.error("Defina TYPESAFE_API_KEY no ambiente; nunca grave a chave no código.")
        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

        with TypeSafeClient() as client:
            response = client.system_one(
                model="jev-1.13.0",
                state={
                    "origem": "relato fictício de exercício",
                    "temperatura_c": 80,
                    "vibracao_mm_s": 5,
                    "probabilidade_sintetica": 0.71,
                    "relato": "Quero que uma pessoa revise este resultado didático.",
                },
                questions={
                    "encaminhamento": Choice(
                        instructions="Escolha encaminhamento didático; não controle máquinas.",
                        criteria={
                            "operacao_normal": "Exibir status sem alerta didático.",
                            "agendar_inspecao": "Exibir sugestão didática de inspeção.",
                            "revisao_humana_imediata": "Encaminhar relato a uma pessoa.",
                        },
                    ),
                    "prioridade": Score(
                        instructions="Prioridade de revisão do relato fictício.",
                        criteria=["Pode esperar", "Revisar em breve", "Revisar agora"],
                    ),
                    "pede_humano": Noul(
                        instructions="O relato pede explicitamente revisão humana?"
                    ),
                },
            )
        payload = response.choices["encaminhamento"].model_dump()
        print("Versão do provedor:", response.model)
        print("Score de prioridade:", response.scores["prioridade"].score)
        print("P(sim à pergunta pede_humano):", response.nouls["pede_humano"].noul)
    answer = parse_choice(payload)
    print(json.dumps(asdict(answer), ensure_ascii=False))
    print(json.dumps(asdict(optional_policy(answer)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
