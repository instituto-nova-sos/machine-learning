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

### Primeiro identifique o Python instalado

Não existe um nome de comando que funcione em toda instalação do Windows.
Para a turma, confirme **Python 3.11 ou 3.12** antes de criar o ambiente:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

Se os dois comandos funcionarem e a versão for a recomendada, execute na raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

`-m venv` manda **esse interpretador** executar o módulo que cria o ambiente.
A instalação e os testes seguintes usam explicitamente o Python da `.venv`.
Não é necessário ativá-la: esse procedimento evita depender do launcher `py`
e de permissão para executar o script `Activate.ps1`.
Essa alternativa é documentada em [venv](https://docs.python.org/3/library/venv.html#how-venvs-work).

### Se na máquina o comando for python3.exe

Sua lembrança é compatível com instalações existentes: a documentação do
[Python 3.11 distribuído pela Microsoft Store](https://docs.python.org/3.11/using/windows.html#the-microsoft-store-package)
descreve nomes como `python.exe`, `python3.exe` e `python3.11.exe`.
Se `python` falhar, confira o candidato antes de criar a pasta:

```powershell
python3.exe --version
python3.exe -c "import sys; print(sys.executable)"
```

**Somente se confirmar a versão desejada**, crie com:

```powershell
python3.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`python3.exe` não é um substituto garantido em todo Windows. Depois da criação,
continue usando `.\.venv\Scripts\python.exe`: chamar `python3.exe` novamente pode
selecionar a instalação global, mesmo quando você pretendia usar o ambiente.

### Por que py -3.11 pode falhar?

`py` é um seletor/gerenciador, diferente do interpretador `python.exe`.
O comando `py -3.11 -m venv .venv` é válido, mas precisa encontrar o comando `py`
e uma instalação 3.11 utilizável. Ele não significa “use qualquer Python 3”.
O [launcher tradicional](https://docs.python.org/3.11/using/windows.html#python-launcher-for-windows)
e o [Python Install Manager atual](https://docs.python.org/3/using/windows.html#basic-use)
têm recursos diferentes. Por isso, deixamos `py` como alternativa, não como
pré-requisito das aulas. A falha relatada na turma não foi reproduzida aqui;
sem a mensagem original, não podemos atribuí-la a uma causa específica.

### Diagnosticar sem adivinhar o caminho

No PowerShell, liste os comandos encontrados com
[Get-Command](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/get-command):

```powershell
Get-Command python, python3, py -All -ErrorAction SilentlyContinue
where.exe python
where.exe python3
```

[where.exe](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/where)
procura executáveis no caminho de busca; use a extensão, pois `where`
no PowerShell também pode designar outro comando. Nome encontrado não basta:
`--version` e `sys.executable` confirmam o interpretador que realmente executou.

| Sintoma | O que conferir |
|---|---|
| `py` não reconhecido | comando ausente ou fora do PATH; use o interpretador confirmado |
| `py -3.11` não encontra Python | 3.11 pode não estar instalado/visível para esse seletor |
| `python` abre a Store ou informa que não foi encontrado | pode ser um alias de instalação, não um runtime funcional |
| Versão diferente de 3.11/3.12 | localize a versão do curso antes de criar a `.venv` |
| Instalação recente não aparece no terminal | reabra o terminal e repita a identificação |

Nos **aliases de execução de aplicativo** do Windows, confira a associação de
`python.exe`/`python3.exe`. Não desative todos os aliases indiscriminadamente:
o Install Manager atual também usa esse mecanismo. A orientação depende do que
foi instalado; veja a [solução de problemas oficial](https://docs.python.org/3/using/windows.html#troubleshooting).

Se já conhece o caminho real do interpretador correto, pode usá-lo diretamente.
O exemplo abaixo tem caminho ilustrativo: substitua pelo caminho da sua máquina.
No PowerShell, `&` executa um caminho entre aspas, como explica a
[documentação de operadores](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_operators#call-operator-).

```powershell
& "C:\caminho\do\Python312\python.exe" --version
& "C:\caminho\do\Python312\python.exe" -m venv .venv
```

### Ativação é opcional

Para usar os exemplos escritos como `python ...`, ative no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

O caminho exibido deve estar dentro da `.venv` deste projeto. Para sair, use
`deactivate`. Se a política bloquear o script, continue com
`.\.venv\Scripts\python.exe` em lugar de `python`, sem alterar a política da máquina.
No Prompt de Comando (`cmd.exe`), o script de ativação é
`.venv\Scripts\activate.bat`; não misture os dois shells.

Pesquisa em fontes oficiais em **16 de outubro de 2026**. Estas instruções foram
revisadas documentalmente; não foram executadas em uma máquina Windows nesta
sessão. CI com Python já provisionado não verifica instalação, PATH ou aliases
nos computadores dos alunos.

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
