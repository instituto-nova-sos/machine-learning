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

## Compatibilidade e validação de manutenção

Use Python 3.11 ou 3.12 em um ambiente virtual novo. Instalar dependências permite
executar as práticas; a confirmação vem de executar os testes e os comandos de
treino, avaliação e inferência nesse ambiente:

```bash
python -m pip check
make validate
make validate-torch  # depois de instalar a etapa PyTorch
```

`make validate` também gera o gráfico imobiliário e verifica links locais e blocos
Python independentes da documentação. `make validate-material` permite repetir só
o treino local necessário ao artefato e essa verificação documental. Nenhum desses
comandos usa Jev live.

Para verificar o piso declarado das dependências diretas da base, crie outro
ambiente Python 3.11 e instale com as
[restrições mínimas](../tests/constraints-min.txt):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c tests/constraints-min.txt -e '.[dev]'
python -m pip check
make validate
```

Faça isso em uma cópia separada se já houver uma `.venv` de trabalho. Essas
restrições são um instrumento de manutenção: não congelam dependências transitivas
nem substituem a instalação usual da turma. O piso PyTorch é verificado à parte
com `torch==2.10.0`; a base continua funcionando sem esse extra.

A [validação contínua](../.github/workflows/validacao.yml) foi configurada para a
base em Linux, macOS e Windows com Python 3.11/3.12, mais um ambiente mínimo em
Linux/3.11. A ponte PyTorch usa jobs CPU separados em Linux, nas duas versões de
Python, com o piso 2.10.0 e uma versão atual permitida pelo projeto. A configuração
usa as ações oficiais [checkout](https://github.com/actions/checkout) e
[setup-python](https://github.com/actions/setup-python).

No Windows, a CI executa comandos Python diretamente, sem exigir GNU Make. O
modo UTF-8 é explícito para conservar os textos em português nos arquivos e nas
saídas capturadas; Matplotlib usa o backend Agg, sem abrir janelas.

Uma matriz configurada não é evidência de execução. Consulte o registro mais
recente em [AGENTS.md](../AGENTS.md) e os resultados da execução no GitHub antes
de afirmar compatibilidade com outro sistema operacional.

Verificação local de 9 de outubro de 2026, em macOS ARM64:

| Ambiente | Resultado da suíte |
|---|---|
| Python 3.11.17; base/dev nos mínimos; PyTorch 2.10.0 | 138 testes aprovados |
| Python 3.12.15; base/dev atuais; PyTorch 2.14.1 | 138 testes aprovados |

Na consolidação inicial, sem PyTorch, Python 3.11 aprovou 115 testes e ignorou
explicitamente o módulo PyTorch. A revisão do PR adicionou 19 casos. O ambiente
mínimo emitiu um aviso de depreciação do SciPy usado
pelo Scikit-learn 1.4, sem falhar. As duas execuções usam CPU; não comprovam
compatibilidade com toda combinação de versões permitidas. A CI do commit
16e7ae7 passou nos 11 jobs configurados, incluindo base Windows/Linux; os checks
do commit de revisão devem ser conferidos separadamente no PR.
