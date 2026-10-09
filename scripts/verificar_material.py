"""Verifica links locais e executa exemplos independentes da continuação.

Execute na raiz depois de gerar equipamento_mlp.json. Cada bloco Python é um
novo processo: não depende de variáveis deixadas por um exemplo anterior. Não
executa blocos shell, --live ou URLs. PyTorch é conferido somente se instalado;
skip explícito não significa validação dessa etapa.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

MODULES = (
    "avaliacao-de-modelos",
    "generalizacao-e-overfitting",
    "neuronio-artificial",
    "redes-neurais",
    "backpropagation",
    "deep-learning",
    "pytorch-na-pratica",
    "inferencia-local",
    "modelos-de-decisao",
    "jev-na-pratica",
    "engenharia-de-ml",
    "projeto-integrador",
    "classificacao",
)


def main() -> int:
    """Confere existência de destinos e executa blocos Python, com timeout 60 s.

    Exige cwd na raiz. Links externos/âncoras não são verificados por este script;
    referências externas são conferidas nas fontes primárias durante autoria.
    Retorna zero se todos os destinos existem e blocos terminam; falha exibe local.
    Scripts de material são fontes do próprio projeto, não código não confiável.
    """
    root = Path.cwd()
    if not (root / "AGENTS.md").exists():
        raise RuntimeError("Execute na raiz do repositório.")
    blocks, skipped, links = 0, 0, 0
    torch_installed = importlib.util.find_spec("torch") is not None
    for module in MODULES:
        for document in (root / module).rglob("*.md"):
            # A matemática antiga contém notação em Python com X abstrato.
            # Só pratica.md anuncia blocos independentes nessa etapa.
            if module == "classificacao" and document.name != "pratica.md":
                continue
            for index, code in enumerate(
                re.findall(r"```python\n(.*?)```", document.read_text(encoding="utf-8"), re.S)
            ):
                if "import torch" in code and not torch_installed:
                    skipped += 1
                    continue
                result = subprocess.run(
                    [sys.executable, "-c", code], text=True, capture_output=True, timeout=60
                )
                if result.returncode:
                    raise RuntimeError(f"Bloco {index} em {document}: {result.stderr}")
                blocks += 1
    for document in root.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(root).parts):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#")[0]
            if path:
                links += 1
                if not (document.parent / path).exists():
                    raise RuntimeError(f"Destino local ausente em {document}: {target}")
    print(f"Blocos Python executados: {blocks}; PyTorch ignorados: {skipped}.")
    print(f"Links locais verificados: {links}; nenhum destino ausente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
