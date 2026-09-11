# Instruções e memória do projeto

Este arquivo é a fonte de verdade para agentes e colaboradores que continuarem o desenvolvimento
do **SOS Capacita — Programação com IA**. Leia-o antes de editar a base e confronte o estado
registrado com a implementação real.

## Objetivo permanente

Construir um repositório educacional rigoroso, em português brasileiro, para os módulos:

- Introdução ao Machine Learning;
- Machine Learning na Prática;
- Introdução ao Deep Learning;
- fundamentos matemáticos de redes neurais.

Princípio pedagógico central:

> Compreender os fundamentos computacionais e matemáticos antes de usar abstrações de alto nível.

O material deve formar desenvolvedores capazes de raciocinar sobre dados, modelos, matemática,
avaliação e sistemas de ML — não apenas operar notebooks, APIs ou chamadas como `fit()`.

## Estado atual

Última atualização registrada: **6 de agosto de 2026**.

A primeira trilha foi implementada, executada e testada:

```text
fundamentos-ia
→ aprendizagem-com-dados
→ matematica-para-ml
→ vetores-e-matrizes
→ regressao-linear
→ otimizacao-e-gradiente
```

Ela contém documentação conceitual, prática, exercícios em três níveis, soluções separadas do
instrutor e referências. Módulos futuros não devem ser considerados concluídos enquanto tiverem
somente planejamento.

| Área | Estado | Evidência |
|---|---|---|
| Configuração e documentação raiz | testado | instalação, lint e comandos validados |
| Fundamentos de IA | revisado | conceitos, prática, exercícios e referências |
| Aprendizagem com dados | testado | geração, partição, escala e validação |
| Matemática para ML | testado | derivadas e checagem numérica |
| Vetores e matrizes | testado | Python puro, NumPy e testes |
| Regressão linear | testado | manual, sklearn, CLI e artefato |
| Otimização e gradiente | testado | treino manual, curva e gradient check |
| Classificação e avaliação | planejado | próxima expansão; não há material placeholder |
| Redes neurais, backpropagation e deep learning | planejado | expansão posterior |
| PyTorch e engenharia de ML completa | planejado | CLI inicial existe; expansão posterior |
| Projeto integrador e rubrica | planejado | após os fundamentos restantes |

Um item só muda para **testado** depois de sua execução ser registrada neste arquivo.

## Arquitetura implementada

- `src/sos_ml/from_scratch/`: implementações transparentes para estudo;
- `src/sos_ml/sklearn_models/`: primeira comparação profissional;
- `src/sos_ml/train.py`, `evaluate.py` e `predict.py`: CLIs;
- `scripts/gerar_dados_imoveis.py`: geração determinística do dataset;
- `scripts/visualizar_regressao.py`: gráfico do ajuste e da perda;
- `tests/`: testes unitários, integração de CLI e gradient check;
- `data/`: documentação e dados sintéticos gerados;
- `artifacts/`: modelo e gráficos produzidos localmente;
- diretórios conceituais: material estudantil e soluções de instrutor;
- `docs/`: jornada, metodologia, ambiente, glossário e referências.

## Decisões que devem ser preservadas

1. Todo conteúdo voltado ao estudante, mensagens de terminal, comentários e exemplos fica em
   português brasileiro. Identificadores podem usar inglês profissional quando isso melhorar a API.
2. Diretórios representam conceitos, não números de aula. Não criar nomes `aula01`, `aula02` etc.
3. Python puro precede NumPy; NumPy precede Scikit-learn ou PyTorch.
4. Notebooks são suplementares. Algoritmos importantes pertencem a módulos reutilizáveis.
5. Atividades obrigatórias devem executar em CPU, sem nuvem, conta paga ou GPU.
6. Dados sintéticos nunca devem ser apresentados como evidência do mercado real.
7. A divisão treino/teste ocorre antes do ajuste de transformações.
8. A padronização é ajustada somente no treino. Na regressão atual, os parâmetros aprendidos no
   espaço padronizado são convertidos para área em m² e preço em reais antes da serialização.
9. O conjunto de teste não deve orientar treinamento ou seleção de hiperparâmetros.
10. Referências não podem ser inventadas. Informação bibliográfica duvidosa deve ser marcada para
    revisão humana.
11. Ética, privacidade, viés, proxies, limitações e uso inadequado devem aparecer ao longo das
    práticas, não apenas em um capítulo isolado.
12. Clareza educacional tem prioridade sobre desempenho de produção nas implementações manuais.
13. Código, configurações, dados e artefatos usam caminhos relativos à raiz. A `.venv` contém
    caminhos absolutos gerados pelo Python, fica fora do Git e deve ser recriada após clone ou
    mudança de diretório.
14. Todo código novo deve funcionar também como documentação didática, seguindo o padrão iniciado
    em `generate_housing_data()` de `sos_ml.data`. Docstrings e comentários devem ser escritos com
    a máxima profundidade útil, para permitir que estudantes estudem o material de forma autônoma,
    sem depender da explicação de um instrutor.

## Documentação didática no código

Ao criar ou alterar código, documente não apenas **o que** ele executa, mas também **por que** cada
decisão existe e como ela se conecta aos conceitos matemáticos, computacionais e de Machine
Learning ensinados pelo projeto. Essa documentação faz parte do conteúdo pedagógico e não deve ser
tratada como acabamento opcional.

Sempre que aplicável:

- use docstrings em módulos, classes e funções para explicar objetivo, parâmetros, tipos, unidades,
  valores padrão, retorno, exceções, efeitos colaterais, pré-condições e limitações;
- descreva o formato e o significado dos dados, incluindo dimensões, colunas, domínio dos valores e
  unidades como metros quadrados, reais, probabilidades ou classes;
- explique fórmulas, variáveis e operações intermediárias, relacionando a notação matemática à
  implementação;
- registre por que sementes, limiares, hiperparâmetros, validações e escolhas algorítmicas foram
  adotados, distinguindo decisões didáticas de requisitos de produção;
- comente etapas cuja finalidade não seja imediatamente evidente para um estudante, inclusive
  preparação de dados, prevenção de vazamento, estabilidade numérica, otimização, avaliação e
  conversões de escala;
- explicite casos-limite, hipóteses, riscos de interpretação e usos inadequados, especialmente em
  exemplos com dados sintéticos, métricas e previsões;
- inclua exemplos pequenos quando eles tornarem o comportamento ou o contrato mais concreto;
- mantenha comentários próximos do trecho explicado e atualize-os junto com o código, evitando
  documentação desatualizada ou que apenas repita literalmente a instrução Python.

A profundidade deve favorecer o estudo autônomo: um aluno deve conseguir acompanhar entradas,
transformações, cálculos, decisões e saídas lendo o código em sequência. Ainda assim, a
documentação deve permanecer tecnicamente precisa e explicar a intenção real do programa, sem
inventar garantias ou ocultar simplificações pedagógicas.

## Última validação conhecida

Ambiente usado: `.venv` local, Python 3.14 disponível na máquina. O projeto declara Python
`>=3.11`; a documentação recomenda 3.11 ou 3.12 para estudantes.

```text
Ruff: aprovado
Mypy: 17 arquivos-fonte, sem erros
Pytest: 18 testes aprovados
Dataset: 120 exemplos sintéticos, semente 42
Treino: 96 exemplos
Teste preservado: 24 exemplos
RMSE observado: aproximadamente R$ 18.603,65
Inferência para 85 m²: aproximadamente R$ 350.577,02
Gráfico: artifacts/regressao_e_perda.png
```

Os valores podem variar se o gerador, a partição, a semente ou o algoritmo forem alterados. Uma
mudança intencional deve atualizar os testes e este registro.

## Validação após clone ou mudança de diretório

Na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy
python -m pytest
python scripts/gerar_dados_imoveis.py
python -m sos_ml.train --data data/processed/imoveis.csv --output artifacts/modelo_linear.json
python -m sos_ml.evaluate --data data/processed/imoveis.csv --model artifacts/modelo_linear.json
python -m sos_ml.predict --model artifacts/modelo_linear.json --area 85
python scripts/visualizar_regressao.py
```

No Windows PowerShell, a ativação é `.venv\Scripts\Activate.ps1`. Se o caminho mudar, recrie a
`.venv`; dados e artefatos ignorados pelo Git podem ser regenerados pelos comandos acima.

## Próxima fase recomendada

Continuar nesta ordem, mantendo cada etapa executável antes de avançar:

1. `classificacao`: sigmoid, limiar, fronteira de decisão, classificador binário simplificado,
   regressão logística manual, NumPy e Scikit-learn, com pequeno projeto de falha de equipamento.
2. `avaliacao-de-modelos`: matriz de confusão, acurácia, precisão, revocação, F1,
   desbalanceamento, baselines e testes das métricas manuais.
3. `generalizacao-e-overfitting`: treino, validação, teste, underfitting, overfitting,
   regularização, validação cruzada, leakage e exemplos deliberadamente enganosos.
4. `neuronio-artificial` e `redes-neurais`: ativações, neurônio, camada densa e MLP em NumPy,
   com testes de forma e forward propagation.
5. `backpropagation`: exemplo escalar completo, rede mínima com intermediários, implementação
   vetorizada e verificações por diferenças finitas.
6. `deep-learning` e `pytorch-na-pratica`: correspondência entre implementações manuais e
   PyTorch; regressão, classificador, MLP, treino, persistência e inferência em CPU.
7. `engenharia-de-ml` e `projeto-integrador`: configuração, logs, esquemas, reprodutibilidade,
   segurança de artefatos, monitoramento, aplicação final, testes, rubrica e análise ética.

Antes de criar novos diretórios, verificar se haverá conteúdo real. Não criar árvores vazias para
simular completude.

## Componentes manuais pendentes

A especificação prevê estes arquivos adicionais em `src/sos_ml/from_scratch/`:

```text
logistic_regression.py
activations.py
neuron.py
dense_layer.py
neural_network.py
backpropagation.py
metrics.py
```

Os arquivos existentes de estatística, vetores, matrizes, regressão, perdas, gradiente, partição e
escala devem ser estendidos somente quando a nova etapa exigir, preservando APIs e testes quando
possível.

Também permanecem pendentes implementações em `torch_models/`, avaliação e classificação
profissionais, projeto integrador, rubrica final e testes de serialização PyTorch.

## Critério para atualizar este arquivo

Ao encerrar cada nova fase:

1. atualizar a tabela de estado sem marcar placeholders como concluídos;
2. registrar novos módulos, decisões e migrações de API;
3. atualizar a contagem e o resultado dos testes;
4. registrar comandos realmente executados;
5. listar falhas ou referências que ainda exijam revisão humana;
6. confirmar que geração, treino, avaliação e inferência continuam reproduzíveis.

## Retomada do trabalho

Ao iniciar em um novo contexto, leia integralmente `AGENTS.md`, `README.md` e
`docs/jornada-de-aprendizado.md`; inspecione e valide o repositório antes de editar. Preserve as
decisões registradas e não refaça trabalho concluído. A próxima fase começa por classificação, com
conteúdo em português brasileiro, implementação manual antes de bibliotecas, testes, referências
verificáveis e execução em CPU.
