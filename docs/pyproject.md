# Entendendo o `pyproject.toml`

[← Voltar à aula principal](../README.md) · [Ambiente local](ambiente-local.md)

## Por que este arquivo aparece neste módulo?

Um script isolado pode ser executado diretamente com `python arquivo.py`. Este projeto já possui
outras necessidades: módulos compartilhados, comandos com `python -m`, bibliotecas externas,
testes e ferramentas de qualidade. Precisamos de uma descrição única e legível por ferramentas.

O `pyproject.toml` é o arquivo moderno de configuração de um projeto Python. Ele não contém o
código do modelo; descreve como preparar, instalar, construir e verificar o projeto.

Sem essa configuração, cada pessoa precisaria descobrir e configurar manualmente:

- quais versões de Python e bibliotecas usar;
- onde encontrar o pacote `sos_ml`;
- como gerar um pacote instalável;
- quais opções passar a pytest, Ruff e mypy;
- quais dependências são necessárias apenas para desenvolvimento.

## O formato TOML

TOML organiza configurações em seções, chaves e valores:

```toml
[uma.secao]
nome = "valor"
ativo = true
itens = ["a", "b"]
```

Colchetes definem uma seção. Strings usam aspas; listas usam colchetes na atribuição. Comentários
começam com `#`.

## Anatomia do arquivo deste projeto

### Sistema de construção

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"
```

Essa seção responde: “qual ferramenta sabe transformar este código em um pacote Python?”. O pip
pode criar um ambiente isolado de construção, instalar o `setuptools` e pedir ao backend que gere
os metadados ou um pacote distribuível.

O **backend de build** não é o compilador do modelo de ML. Ele é a ferramenta de empacotamento do
projeto Python.

### Metadados do projeto

```toml
[project]
name = "sos-capacita-machine-learning"
version = "0.1.0"
requires-python = ">=3.11"
```

- `name` é o nome usado na instalação e distribuição;
- `version` identifica a evolução publicada do projeto;
- `requires-python` impede instalação em versões incompatíveis;
- `description`, `readme` e `license` documentam o pacote.

O nome de distribuição pode conter hífens, enquanto o pacote importado usa um identificador Python:

```text
distribuição: sos-capacita-machine-learning
importação:   sos_ml
```

### Dependências de execução

```toml
dependencies = [
  "numpy>=1.26,<3",
  "pandas>=2.1,<3",
  "matplotlib>=3.8,<4",
  "scikit-learn>=1.4,<2",
]
```

Essas bibliotecas são necessárias para executar as partes correspondentes do material. Cada
intervalo aceita atualizações compatíveis dentro de um limite conhecido. Fixar exatamente todas
as versões aumenta a reprodutibilidade imediata, mas exige manutenção frequente; intervalos
controlados equilibram compatibilidade e atualização neste curso.

### Dependências opcionais

```toml
[project.optional-dependencies]
dev = ["pytest>=8,<9", "ruff>=0.6,<1", "mypy>=1.11,<2"]
torch = ["torch>=2.2,<3"]
```

Grupos opcionais evitam instalar tudo para todos os usos:

```bash
# Projeto com ferramentas de desenvolvimento
python -m pip install -e ".[dev]"

# Projeto com PyTorch, quando a trilha chegar a essa etapa
python -m pip install -e ".[torch]"

# Ambos os grupos
python -m pip install -e ".[dev,torch]"
```

PyTorch não faz parte das atividades iniciais obrigatórias e pode ser uma instalação grande; por
isso está separado.

### Layout `src`

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

O código importável fica em `src/sos_ml`, em vez de ser misturado à raiz. Esse layout reduz o
risco de os testes importarem acidentalmente arquivos locais que não fariam parte de uma
instalação real.

O arquivo `src/sos_ml/py.typed` é incluído como dado do pacote para indicar que a biblioteca
fornece informações de tipos.

### Configuração das ferramentas

Seções iniciadas por `[tool.]` pertencem a ferramentas específicas:

- `[tool.pytest.ini_options]`: localização e opções dos testes;
- `[tool.ruff]` e `[tool.ruff.lint]`: versão-alvo, tamanho de linha e regras de lint;
- `[tool.mypy]`: versão de Python simulada e rigor da análise de tipos.

Centralizar essas opções evita arquivos separados para cada ferramenta e faz com que os mesmos
comandos funcionem em diferentes máquinas a partir da raiz do repositório.

## O que o comando de instalação faz?

```bash
python -m pip install -e ".[dev]"
```

Parte por parte:

| Trecho | Significado |
|---|---|
| `python -m pip` | executa o pip pertencente ao Python ativo |
| `install` | solicita uma instalação |
| `-e` | usa modo editável |
| `.` | instala o projeto do diretório atual |
| `[dev]` | inclui o grupo opcional de desenvolvimento |

No modo editável, alterações em `src/sos_ml` ficam disponíveis sem reinstalar o pacote a cada
edição. Alterações nas dependências ou nos metadados do `pyproject.toml` podem exigir que o
comando seja executado novamente.

Usar `python -m pip` é preferível a chamar apenas `pip`, pois reduz a chance de instalar uma
biblioteca em outro Python fora da `.venv` ativa.

## Comandos do dia a dia

```bash
# Executar os testes configurados no pyproject.toml
python -m pytest

# Verificar estilo e possíveis erros
python -m ruff check .

# Verificar anotações de tipo
python -m mypy

# Corrigir formatação
python -m ruff format .

# Construir wheel e pacote-fonte, se a ferramenta build estiver instalada
python -m build
```

Construir um pacote não é necessário para acompanhar as aulas. O comando é mostrado para separar
dois conceitos: **instalação editável**, usada no desenvolvimento, e **artefato de distribuição**,
usado para publicar ou instalar uma versão fechada.

## Quando editar o `pyproject.toml`?

Edite-o quando houver uma mudança real no contrato do projeto, por exemplo:

- adicionar uma dependência usada pelo código;
- aumentar a versão mínima do Python;
- criar um novo grupo opcional;
- ajustar regras compartilhadas de testes, lint ou tipos;
- alterar versão, licença ou metadados do pacote.

Não adicione uma biblioteca apenas porque ela foi experimentada localmente. Dependências aumentam
tempo de instalação, superfície de segurança e possibilidade de incompatibilidade.

## Erros frequentes

### `ModuleNotFoundError: No module named 'sos_ml'`

Confirme que a `.venv` está ativa e execute novamente:

```bash
python -m pip install -e ".[dev]"
```

### O pip instalou no lugar errado

Confira os executáveis ativos:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Ambos devem apontar para a `.venv` do projeto.

### O projeto foi movido e deixou de importar

Ambientes virtuais guardam caminhos absolutos. Não edite manualmente os arquivos internos da
`.venv`; recrie o ambiente no novo local e reinstale o projeto.

---

[← Voltar à aula principal](../README.md) · [Próximo: jornada de aprendizado →](jornada-de-aprendizado.md)
