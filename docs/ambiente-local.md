# Ambiente local

[← Voltar à aula principal](../README.md) · [Entendendo o `pyproject.toml` →](pyproject.md)

## Clone novo ou projeto movido

O ambiente virtual não faz parte do projeto portátil. O Python grava nele caminhos absolutos da
máquina que o criou, por isso `.venv/` está no `.gitignore`. Sempre crie uma nova `.venv` após
clonar, baixar ou mover esta pasta; não copie um ambiente criado em outro diretório ou computador.

## Linux e macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Se a política bloquear scripts, use uma sessão PowerShell comum e execute
`Set-ExecutionPolicy -Scope Process RemoteSigned`; a alteração vale apenas para aquela sessão.
Para sair do ambiente, execute `deactivate`. Não instale dependências globalmente.
