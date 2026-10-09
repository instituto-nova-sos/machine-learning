"""Projeto local: treino separado da inferência, decisão, política e observabilidade."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .decisions import EquipmentState, LocalDecisionModel, evaluate_decision
from .equipment_io import load_equipment_model, save_equipment_model
from .neural_training import train_equipment


def main(argv: list[str] | None = None) -> int:
    """CLI CPU com artefato relativo, saída JSON e registro opcional JSONL.

    treinar executa desenvolvimento/avaliação e persiste; inferir somente lê
    artefato e faz forward. Arquivo de log, quando solicitado, recebe uma linha
    por chamada com resultado/motivo/versão; não guarda sensores por padrão.
    --measure mede pipeline aquecido de 100 entradas iguais, sem custo de carga.
    """
    parser = argparse.ArgumentParser(description="IA local educacional; equipamentos sintéticos.")
    commands = parser.add_subparsers(dest="mode", required=True)
    train_parser = commands.add_parser("treinar", help="Treina e salva MLP pequena.")
    train_parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    train_parser.add_argument("--output", type=Path, default=Path("artifacts/equipamento_mlp.json"))
    train_parser.add_argument("--plot", type=Path)
    predict_parser = commands.add_parser("inferir", help="Carrega modelo sem retreinar.")
    predict_parser.add_argument(
        "--model", type=Path, default=Path("artifacts/equipamento_mlp.json")
    )
    predict_parser.add_argument(
        "--temperature", type=float, required=True, help="Temperatura em °C."
    )
    predict_parser.add_argument("--vibration", type=float, required=True, help="Vibração em mm/s.")
    predict_parser.add_argument(
        "--high-consequence", action="store_true", help="Exige revisão humana."
    )
    predict_parser.add_argument("--log", type=Path, help="Registro JSONL opcional.")
    predict_parser.add_argument(
        "--measure", action="store_true", help="Mede 100 chamadas aquecidas."
    )
    args = parser.parse_args(argv)
    if args.mode == "treinar":
        result = train_equipment(args.backend)
        save_equipment_model(result.artifact, args.output)
        print(
            f"Melhor época pela validação: {result.best_epoch}; teste final: {result.test_metrics}"
        )
        print(f"Baseline final: {result.baseline_metrics}")
        print(f"Perda treino: {result.training_loss[0]:.6f} → {result.training_loss[-1]:.6f}")
        print(f"Artefato: {args.output}; tamanho: {args.output.stat().st_size} bytes.")
        if args.plot:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots()
            axis.plot(result.training_loss, label="Treino")
            axis.plot(result.validation_loss, label="Validação")
            axis.axvline(result.best_epoch, color="gray", linestyle="--", label="Estado restaurado")
            axis.set(
                xlabel="Época (zero = antes do treino)",
                ylabel="Entropia cruzada média",
                title="MLP — equipamentos sintéticos",
            )
            axis.legend()
            figure.tight_layout()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.plot)
            plt.close(figure)
    else:
        try:
            state = EquipmentState(args.temperature, args.vibration, args.high_consequence)
        except ValueError as error:
            parser.error(str(error))
        try:
            artifact = load_equipment_model(args.model)
        except (ValueError, OSError) as error:
            parser.error(f"Não foi possível carregar o artefato: {error}")
        provider = LocalDecisionModel(artifact)
        decision, policy = evaluate_decision(provider, state)
        report = {
            "versao": 1,
            "sintetico": True,
            "artefato_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "decisao": asdict(decision),
            "politica": asdict(policy),
        }
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        if args.measure:
            # A chamada acima aquece o caminho. Mede transformação+inferência+política;
            # não mede disco/carga, tela, rede ou inicialização do processo.
            start = perf_counter()
            for _ in range(100):
                evaluate_decision(provider, state)
            elapsed = (perf_counter() - start) / 100
            count = sum(
                p.size
                for p in (
                    artifact.model.hidden.weights,
                    artifact.model.hidden.bias,
                    artifact.model.output.weights,
                    artifact.model.output.bias,
                )
            )
            print(f"CPU NumPy: {count} parâmetros float64 ({count * 8} bytes de parâmetros).")
            print(f"Pipeline aquecido: média {elapsed * 1000:.4f} ms/entrada em 100 chamadas.")
            print("Não inclui runtime, carga do arquivo ou avaliação de precisão.")
    print("Demonstração sintética: não controla máquinas nem fornece conselho industrial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
