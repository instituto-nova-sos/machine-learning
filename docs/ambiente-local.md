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

## Etapas PyTorch e Jev

O grupo `torch` é instalado quando você chega à prática PyTorch; consulte
[instalação CPU e compatibilidade](../pytorch-na-pratica/README.md). A instalação
base continua leve em relação a frameworks; não baixa modelos pré-treinados.
Use Python 3.11/3.12 para a turma e confira wheels no seletor oficial. O ambiente
registrado em AGENTS pode usar versão mais nova e não valida todos os sistemas.

`make validate` executa caminhos locais com NumPy; se torch estiver instalado,
pytest também executa seus testes. Sem ele, o arquivo de testes PyTorch é ignorado
explicitamente. `make validate-torch` **exige** instalação e verifica a etapa.
Não interprete um skip como PyTorch validado.

`jev` é integração remota opcional. `python scripts/jev_opcional.py` usa fixture
local e funciona com a base. Para o exercício remoto, leia
[Jev na prática](../jev-na-pratica/README.md); nunca grave credenciais no Git.
`make validate` não importa esse SDK nem consulta o serviço oficial.
