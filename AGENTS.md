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

Última atualização registrada: **9 de outubro de 2026**.

A trilha foi implementada até o projeto integrador, com execução local das práticas:

```text
fundamentos-ia
→ aprendizagem-com-dados
→ matematica-para-ml
→ vetores-e-matrizes
→ regressao-linear
→ otimizacao-e-gradiente
→ classificacao
→ avaliacao-de-modelos → generalizacao-e-overfitting
→ neuronio-artificial → redes-neurais → backpropagation
→ deep-learning → pytorch-na-pratica → inferencia-local
→ modelos-de-decisao → jev-na-pratica → engenharia-de-ml
→ projeto-integrador
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
| Classificação | testado | sigmoid, limiar, logística manual/NumPy/sklearn e projeto sintético |
| Avaliação de modelos | testado | métricas manuais e matriz da logística anterior |
| Generalização e sobreajuste | testado | partição tripla e curva de capacidade/validação |
| Neurônio e redes neurais | testado | Python puro, camadas/MLP NumPy e formas |
| Retropropagação | testado | cadeia escalar, gradientes e check de todos os parâmetros |
| Deep Learning | revisado | conceitos e contas de CNN/RNN/atenção; sem treino grande |
| PyTorch | testado | equivalência, CPU, SGD, state_dict e inferência em processo novo |
| Inferência local | testado | JSON validado, escalas fixas, latência e contagem de parâmetros |
| Decisões e revisão humana | testado | contratos, abstenção, política e fallback |
| Jev / System One | revisado e testado offline | fontes oficiais e fixture; integração live não executada |
| Engenharia de ML educacional | testado | contrato de artefato, CLI e registro JSONL; não é produção completa |
| Projeto integrador e rubrica | testado | fluxo local completo e rubrica de 100 pontos |

Um item só muda para **testado** depois de sua execução ser registrada neste arquivo.

## Última validação: consolidação — 9 de outubro de 2026

Os testes foram executados em macOS ARM64 com ambientes separados do ambiente de trabalho:

| Python | Dependências diretas da base | PyTorch CPU | Suíte final |
|---|---|---|---|
| 3.11.17 | NumPy 1.26.0, pandas 2.1.0, Matplotlib 3.8.0, Scikit-learn 1.4.0 | 2.10.0 | 119 aprovados |
| 3.12.15 | NumPy 2.5.3, pandas 2.3.3, Matplotlib 3.11.2, Scikit-learn 1.9.1 | 2.14.1 | 119 aprovados |

Ruff 0.6.0 e mypy 1.11.0 passaram no ambiente mínimo; Ruff 0.17.0 e mypy 1.20.2
passaram no ambiente atual. Mypy verificou 39 fontes em ambos. O teste inicial da
base sem PyTorch em Python 3.11 teve 115 aprovados e um módulo ignorado; a suíte
de 119 foi executada depois de instalar a etapa PyTorch. Não confundir esse skip
inicial com validação do extra.

O ambiente mínimo apresentou um aviso de depreciação de opções L-BFGS-B no
Scikit-learn 1.4.0 com SciPy 1.16.3, sem falha. As restrições mínimas abrangem as
dependências diretas da base e do grupo dev, não todas as transitivas ou o backend
de construção. Linux/Windows ainda dependem da execução da CI configurada;
configuração e parse de YAML não equivalem a executar esses sistemas.

Os registros de 2 de outubro abaixo são históricos. Consulte também o registro
detalhado desta consolidação no final do arquivo.

## Arquitetura implementada

- `src/sos_ml/from_scratch/`: implementações transparentes para estudo;
- `src/sos_ml/sklearn_models/`: primeira comparação profissional;
- `src/sos_ml/train.py`, `evaluate.py` e `predict.py`: CLIs;
- `scripts/gerar_dados_imoveis.py`: geração determinística do dataset;
- `scripts/visualizar_regressao.py`: gráfico do ajuste e da perda;
- `src/sos_ml/from_scratch/activations.py`: sigmoid estável e decisão por limiar;
- `src/sos_ml/from_scratch/logistic_regression.py`: logística binária em Python puro;
- `src/sos_ml/from_scratch/logistic_regression_numpy.py`: operações vetorizadas com API equivalente;
- `src/sos_ml/sklearn_models/logistic_regression.py`: comparação sem penalização;
- `src/sos_ml/equipment_data.py`: geração sintética e preparação sem vazamento;
- `src/sos_ml/classify.py`: experimento completo e gráfico opcional de treino;
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

## Histórico: validação até classificação

Ambiente usado: `.venv` local, Python 3.14 disponível na máquina. O projeto declara Python
`>=3.11`; a documentação recomenda 3.11 ou 3.12 para estudantes.

```text
Ruff: aprovado
Mypy: 23 arquivos-fonte, sem erros
Pytest: 76 testes aprovados
Dataset: 120 exemplos sintéticos, semente 42
Treino: 96 exemplos
Teste preservado: 24 exemplos
RMSE observado: aproximadamente R$ 18.603,65
Inferência para 85 m²: aproximadamente R$ 350.577,02
Gráfico: artifacts/regressao_e_perda.png
```

Classificação executada em CPU com Python 3.14.7 e Scikit-learn 1.9.0:

```text
Dataset: 400 exemplos sintéticos; semente 42
Partição: semente 17, 320 exemplos de treino e 80 de teste
Falhas observadas: 118 no treino e 33 no teste
Treino manual e NumPy: taxa 0,1; 2000 épocas
Perda manual após primeira/final atualização: 0,683351 → 0,442527
Pesos padronizados: aproximadamente [1,357201; 1,333977]; viés -0,890922
Cada implementação: 64/80 acertos; 6 falsos alertas; 10 falhas não detectadas
Baseline de classe majoritária no treino: classe 0, 47/80 acertos
Nova medida (80 °C, 5 mm/s): probabilidade estimada 0,5696
Gráfico: artifacts/classificacao_e_perda.png
```

Decisões da classificação: dados independentes com duas colunas (temperatura, vibração), alvo de
falha nas próximas 24 horas, rótulos Bernoulli com sobreposição e scalers por coluna ajustados
apenas no treino. As constantes do gerador definem um mundo artificial, não estatísticas
aprendidas. O limiar padrão 0,5 inclui o empate na classe 1. A versão NumPy herda o ciclo manual
e vetoriza escores, probabilidades, perda e gradientes; mantém listas na API por clareza didática.
Scikit-learn usa C infinito e L-BFGS para comparar o objetivo sem regularização. Não há migração
de APIs existentes nem persistência do classificador. Métricas completas, seleção de limiar em
validação e uso operacional permanecem fora desta etapa.

Comandos efetivamente executados nesta fase, na raiz e usando a `.venv` existente:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest
PATH="$PWD/.venv/bin:$PATH" make validate
.venv/bin/python scripts/visualizar_regressao.py
.venv/bin/python -m sos_ml.classify --plot artifacts/classificacao_e_perda.png
git diff --check
```

`make validate` executou também geração, treino, avaliação e inferência imobiliária e confirmou
os resultados anteriores. Os três blocos Python de `classificacao/pratica.md` foram extraídos e
executados separadamente; os 29 links locais do novo módulo foram conferidos. O gráfico de
classificação foi inspecionado visualmente. A revisão local verificou derivadas, estabilidade,
equivalência das implementações e prevenção de vazamento. Sem falhas pendentes ou referências
duvidosas identificadas. No sandbox, Matplotlib/Fontconfig emitiram avisos de cache sem permissão;
o fallback temporário permitiu gerar o gráfico de classificação. O script anterior de gráfico
imobiliário abortou no sandbox e foi reexecutado com permissão fora dele, com sucesso.
Python 3.11/3.12 e versões mínimas das
dependências não foram executados nesta fase; a validação registrada refere-se ao ambiente local.

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
python -m sos_ml.classify --plot artifacts/classificacao_e_perda.png
```

No Windows PowerShell, a ativação é `.venv\Scripts\Activate.ps1`. Se o caminho mudar, recrie a
`.venv`; dados e artefatos ignorados pelo Git podem ser regenerados pelos comandos acima.

## Histórico: planejamento anterior à expansão de outubro

O roteiro abaixo orientou a expansão registrada depois. As etapas foram implementadas;
consulte a tabela atual e os registros de execução, sem repetir trabalho concluído:

1. `avaliacao-de-modelos`: matriz de confusão, acurácia, precisão, revocação, F1,
   desbalanceamento, baselines e testes das métricas manuais.
2. `generalizacao-e-overfitting`: treino, validação, teste, underfitting, overfitting,
   regularização, validação cruzada, leakage e exemplos deliberadamente enganosos.
3. `neuronio-artificial` e `redes-neurais`: ativações, neurônio, camada densa e MLP em NumPy,
   com testes de forma e forward propagation.
4. `backpropagation`: exemplo escalar completo, rede mínima com intermediários, implementação
   vetorizada e verificações por diferenças finitas.
5. `deep-learning` e `pytorch-na-pratica`: correspondência entre implementações manuais e
   PyTorch; regressão, classificador, MLP, treino, persistência e inferência em CPU.
6. `engenharia-de-ml` e `projeto-integrador`: configuração, logs, esquemas, reprodutibilidade,
   segurança de artefatos, monitoramento, aplicação final, testes, rubrica e análise ética.

Antes de criar novos diretórios, verificar se haverá conteúdo real. Não criar árvores vazias para
simular completude.

## Componentes manuais previstos e agora implementados

A expansão implementou estes arquivos adicionais em `src/sos_ml/from_scratch/`:

```text
neuron.py
dense_layer.py
neural_network.py
backpropagation.py
metrics.py
```

Os arquivos existentes de estatística, vetores, matrizes, regressão, perdas, gradiente, partição e
escala devem ser estendidos somente quando a nova etapa exigir, preservando APIs e testes quando
possível.

A expansão acrescenta `torch_models/equipment.py`, métricas, MLP, treinamento com
validação, artefatos de inferência, decisões e projeto integrador. Não inclui
serviço de produção, implantação ExecuTorch, redes grandes ou controle industrial.
Esses limites são explícitos, não módulos vazios. A logística anterior permanece
sem alterações de API e sem persistência própria; a nova persistência é da MLP.

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
decisões registradas e não refaça trabalho concluído. A continuação até projeto integrador está implementada; leia também
`docs/plano-evolucao.md` e os registros abaixo. Preserve português brasileiro,
implementação manual antes de bibliotecas, referências verificáveis e CPU.
Futuras extensões devem ser motivadas por uma tarefa concreta; não refaça estas fases.

## Evolução: Avaliação de modelos — 2 de outubro de 2026

Contagens manuais, matriz, acurácia, precisão, revocação, especificidade, F1 e baseline com métricas indefinidas explícitas; APIs anteriores preservadas.

Testado: make validate com MPLBACKEND=Agg e caches temporários: Ruff aprovado, mypy 25 fontes, 81 testes aprovados. python -m sos_ml.assess: matriz [[41,6],[10,23]], acurácia 0,8 e F1 0,741935. Regressão RMSE 18603,65 e inferência 350577,02 reproduzidas. Classificação anterior reproduzida. Uma execução simultânea do mypy falhou internamente; execução isolada com --cache-dir=/tmp/sos-mypy-phase1 passou.

## Evolução: Generalização e sobreajuste — 2 de outubro de 2026

Partição manual 60/20/20; capacidade, regularização, validação cruzada e experimento deliberado com árvore no mesmo domínio sintético.

Testado: python -m sos_ml.generalization --plot artifacts/generalizacao.png; pytest tests/test_generalization.py (2 aprovados), mypy (26 fontes) e Ruff aprovados. Profundidade 7 escolhida por validação: erro 0,2625; teste final 0,30. Profundidade 12: treino 0 e validação 0,35. Gráfico gerado e inspecionado.

## Evolução: Neurônio artificial — 2 de outubro de 2026

Soma ponderada e ativações linear, sigmoid e ReLU em Python puro; equivalência explícita com logística e composição de camadas afins.

Testado: pytest tests/test_neuron.py (4 aprovados), Ruff e mypy (27 fontes) aprovados. Cálculo manual z=0,8 e equivalência sigmoid/logística verificados nos testes.

## Evolução: Redes neurais — 2 de outubro de 2026

Camada densa e MLP NumPy 2→4→1; formas explícitas e propagação para frente com saída em logits.

Pytest tests/test_network.py: 2 aprovados; Ruff aprovado. Mypy inicialmente encontrou alias NumPy sem TypeAlias explícito; correção e nova validação registradas abaixo. Forward manual da camada produz [5,2]; rede do exemplo produz logit 3.

## Evolução: Retropropagação — 2 de outubro de 2026

Cadeia escalar em Python puro, gradientes vetorizados da MLP, atualização separada e checagem de todos os parâmetros por diferenças centrais.

Executados: network_demo, pytest tests/test_backpropagation.py (3 aprovados), Ruff e mypy. Erro máximo do gradient check 4,31e-11; perda de treino NumPy 0,765569→0,441734 em 500 épocas. Mypy exigiu anotação explícita de labels, corrigida antes da nova execução. Alias NumPy da fase anterior corrigido: 29 fontes sem erros.

## Evolução: Deep Learning — 2 de outubro de 2026

Aprendizado de representações, profundidade, CNN/RNN/Transformer e distinção arquitetura/tarefa, com cálculos pequenos locais.

Módulo conceitual revisado; não há treinamento de modelos grandes. network_demo executado na fase de retropropagação; exercícios de filtro, recorrência e atenção têm soluções separadas.

## Evolução: PyTorch na prática — 2 de outubro de 2026

Tensores CPU, autograd, Module/Parameter/Linear, BCE, SGD, modos, state_dict, persistência e inferência em outro processo; extra torch preservado.

Testado com torch 2.14.1 e Python 3.14.7: 8 testes de PyTorch/persistência aprovados; Ruff e mypy (36 fontes) aprovados. torch_demo treinar e inferir executados em processos separados: época 92 restaurada, acurácia 0,775, F1 0,7; p(80 °C,5 mm/s)=0,651786. BCE treino 0,763558→0,422299 (última época antes da restauração). Artefatos .pt weights_only e JSON comparados. Python 3.11/3.12 e versões mínimas não executados.

## Evolução: Inferência local — 2 de outubro de 2026

Persistência JSON com arquitetura, parâmetros e scalers; processo separado, CPU, medição de bytes/latência e discussão de quantização/ExecuTorch.

Testado: local_ai treinar --plot artifacts/mlp_treino_validacao.png; inferir --temperature 80 --vibration 5 --measure --log artifacts/decisoes.jsonl. Artefato JSON 1006 bytes, 17 parâmetros (136 bytes float64); média aquecida observada 0,0214 ms em 100 chamadas, sem carga/disco. Não é benchmark portátil. PNG de treino/validação inspecionado. Pytest de persistência e CLI aprovado.

## Evolução: Modelos de decisão e revisão humana — 2 de outubro de 2026

Fronteira local tipada, espaço limitado, política determinística, concentração binária explicitamente definida, abstenção e comparação classificador/decisão/LLM.

Testado: pytest tests/test_decisions.py tests/test_local_ai_cli.py (10 aprovados), Ruff e mypy (38 fontes). Consequência elevada em 80/5 exige humano; 150/5 abstém antes de inferir; falha de provedor mockado exige revisão. CLI/log JSONL em novo processo preservam artefato. Nenhuma saída controla máquina real.

## Evolução: Jev como estudo de caso — 2 de outubro de 2026

Choice/Score/Noul verificados na documentação oficial; camada local obrigatória e script remoto opcional separado, sem keys ou chamadas obrigatórias.

Executados: scripts/jev_opcional.py em modo offline e pytest tests/test_jev_boundary.py (6 aprovados). Fixture inventado validado e encaminhado a humano; nenhuma credencial ou chamada real utilizada. Extra jev fixa SDK oficial 0.7.2 conforme pyproject publicado, Python >=3.10 e MIT do SDK. Ruff e mypy corrigidos antes da validação consolidada; integração live não foi executada.

## Evolução: Engenharia de ML — 2 de outubro de 2026

Contratos versionados, pré-processamento persistido, observabilidade JSONL, reprodutibilidade, segurança de artefatos e responsabilidade humana.

Exercícios apoiados pelos testes de persistência, CLI e política; make validate executado com 116 testes aprovados, Ruff e mypy 39 fontes. Log sintético inspecionado; não existe controle de equipamento ou serviço de produção.

## Evolução: Projeto integrador — 2 de outubro de 2026

Fluxo local completo de sensores sintéticos até modelo, artefato, inferência, decisão, política, revisão e registro; rubrica de 100 pontos e soluções separadas.

make validate executou regressão/classificação antigas, métricas, sobreajuste, gradient check, treino/persistência e inferência. 116 testes aprovados. Caminho PyTorch também executado em processos separados, com resultados equivalentes. Jev oficial permanece opcional e não foi chamado.

## Registro consolidado da expansão — 2 de outubro de 2026

A trilha continua até avaliação, generalização, neurônio, redes, retropropagação,
Deep Learning, PyTorch, inferência local, decisões, Jev, engenharia e projeto final.
Cada novo diretório contém conteúdo real, exercícios em três níveis, referências
e soluções separadas. `docs/plano-evolucao.md` registra estado encontrado e plano.
A tabela no início deste arquivo descreve o estado atual; os registros anteriores
são históricos e não indicam que a continuação ainda esteja planejada.

Novas APIs: ConfusionCounts/majority_class; partição tripla;
Neuron/DenseLayer/BinaryMLP; gradientes/check/fit separados;
EquipmentDevelopmentSplit; EquipmentArtifact e JSON versão 1;
EquipmentMLP e state_dict; TrainingResult com seleção/restauração;
EquipmentState/StructuredDecision/DecisionModel/PolicyResult;
validação offline JevChoice. Não houve migração de CLIs imobiliárias/logística.

Decisões de implementação:

- 400 exemplos do mesmo gerador sintético; MLP 2→4→1, 17 parâmetros float64;
- 240/80/80 antes de escala; taxa 0,1 e parada pela validação (paciência 100);
- melhor estado restaurado, inclusive estado inicial se nenhuma época melhorar;
- teste não orienta parâmetros/parada; teste já publicado na trilha é reaproveitado
  didaticamente e não anunciado como benchmark independente;
- PyTorch opcional por etapa, base NumPy preservada; extra torch elevado de
  >=2.2 para >=2.10,<3 devido ao aviso oficial GHSA-63cw-57p8-fm3p;
- Python >=3.11 preservado, recomendação estudantil 3.11/3.12 preservada;
- Jev SDK 0.7.2 em extra opcional; fontes oficiais consultadas, sem chave/chamada live;
- concentração binária local |2p−1|, não probabilidade de acerto de ação;
- política verifica domínio/consequência/ambiguidade e jamais controla máquinas;
- JSONL inclui decisão, motivo e identificador SHA-256 do artefato, sem sensores brutos;
- .pt/JSONL/.env ignorados; JSON e state_dict validados; origem ainda deve ser confiável.

Ambiente efetivamente executado: macOS ARM64, Python 3.14.7, NumPy 2.5.3,
Scikit-learn 1.9.1, PyTorch 2.14.1. Wheels e Python 3.11/3.12/versões mínimas
não foram executados. A matriz oficial de versões foi consultada; isso não
substitui executar esses ambientes. A instalação base inicialmente falhou por
DNS no sandbox e foi concluída após aprovação. Não havia .venv nesta cópia.
Uma execução concorrente do mypy falhou internamente; execuções isoladas passaram.
Aliases NumPy e linhas longas novos foram corrigidos antes da validação final.
O verificador documental inicialmente tentou executar notação abstrata antiga;
agora executa os blocos anunciados como independentes, inclusive pratica.md da
classificação, e não pseudocódigo algébrico sem entradas.

Resultados verificados:

- Ruff aprovado; mypy: 39 fontes, sem erros;
- pytest: **119 testes aprovados**, incluindo PyTorch instalado;
- 12 blocos Python executados (9 novos e 3 da classificação anterior);
- 326 links locais conferidos na revisão final, sem destinos ausentes;
- gráfico de sobreajuste e gráfico MLP treino/validação inspecionados visualmente;
- matriz logística [[41,6],[10,23]], acurácia 0,8, F1 0,741935;
- árvore: profundidade 7 selecionada, validação 0,2625, teste 0,30;
  profundidade 12: erro treino 0, validação 0,35;
- gradient check de todos os parâmetros: erro máximo 4,31e-11;
- network_demo: BCE treino 0,765569→0,441734 em 500 épocas sem seleção;
- MLP com validação: melhor época 92, acurácia 0,775 e F1 0,7;
  baseline 0,5875, p(80 °C,5 mm/s)=0,651786;
- história de treino 0,763558→0,422299 até época 192, com restauração da 92;
- arquivo JSON 1006 bytes; 136 bytes de parâmetros; pipeline aquecido aproximadamente
  0,02 ms/entrada nesta máquina, sem carga/disco/runtime inicial, não promessa portátil;
- regressão anterior: RMSE 18603,65 e previsão 350577,02 preservados;
- classificação anterior: 64/80 acertos, 6 FP e 10 FN preservados;
- consequência alta exige humano; fora de domínio abstém sem probabilidade;
- teste dedicado confirma que mudar só y do teste não altera pesos/seleção/histórias;
- teste dedicado confirma restauração do melhor estado de validação;
- Jev: fixture inventado validado offline, sem afirmar resultado real do provedor.

Comandos realmente executados (na raiz, com .venv):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install 'torch>=2.2,<3'
.venv/bin/python -m pip install --no-deps --no-build-isolation -e '.[dev]'
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-mpl XDG_CACHE_HOME=/tmp/sos-ml-cache PATH="$PWD/.venv/bin:$PATH" make validate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-mpl XDG_CACHE_HOME=/tmp/sos-ml-cache PATH="$PWD/.venv/bin:$PATH" make validate-torch
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-mpl XDG_CACHE_HOME=/tmp/sos-ml-cache .venv/bin/python scripts/verificar_material.py
.venv/bin/python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --measure --log artifacts/decisoes.jsonl
.venv/bin/python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --high-consequence
.venv/bin/python -m sos_ml.local_ai inferir --temperature 150 --vibration 5
.venv/bin/python scripts/jev_opcional.py
git diff --check
```

O comando de instalação torch anterior ao ajuste de mínimo resolveu 2.14.1,
que satisfaz o requisito final >=2.10. make validate-torch executou quatro testes
de ponte e os dois processos torch_demo. make validate executou toda a suíte,
os comandos antigos e as práticas locais novas com 116 testes naquele momento;
a execução posterior de pytest aprovou os 119 testes da versão final.
Os caches Matplotlib/Fontconfig
foram direcionados a /tmp; nenhuma GPU ou chamada remota foi usada nas práticas.

Limitações abertas: SDK Jev/live não executado com credenciais; não há reprodução
local de pesos oficiais; versões mínimas e sistemas da turma não executados;
nenhum serviço industrial, CNN/RNN/Transformer grande ou ExecuTorch foi implantado.
Essas atividades não são prometidas pela trilha obrigatória. Sem falhas conhecidas
nos caminhos executados ou referência bibliográfica inventada identificada.

## Registro detalhado da consolidação — 9 de outubro de 2026

A expansão encontrada na árvore de trabalho foi preservada na branch
`chore/consolidacao-trilha-validacao-python`, no commit `818f08d`. Essa etapa
registra os 19 módulos existentes, sem nova fase pedagógica ou migração de APIs.
A recomendação de Python 3.11/3.12 foi verificada localmente nos dois ambientes da
tabela de última validação, separados da `.venv` de trabalho.

Mudanças de manutenção:

- `tests/constraints-min.txt` fixa o piso das dependências diretas da base/dev em
  Python 3.11; as transitivas continuam resolvidas, sem promessa de lock completo;
- Ruff 0.6.0 exigiu `int | float` nos testes de tipo do adaptador Jev; mypy 1.11.0
  exigiu `TypeGuard[Action]` explícito. O contrato e as decisões permanecem iguais.
  Um cast intermediário foi rejeitado como redundante pelo mypy atual e substituído
  pelo guard antes da validação final;
- o verificador documental lê Markdown explicitamente como UTF-8;
- `make validate` inclui o gráfico imobiliário e os exemplos/links documentais;
  `validate-material` depende do treino que produz seu artefato, inclusive após clone;
- `.github/workflows/validacao.yml` configura sete jobs da base (três sistemas,
  duas versões Python e um mínimo Linux/3.11) e quatro jobs PyTorch CPU em Linux
  (duas versões Python, piso e atual). YAML analisado localmente; CI remota pendente;
- README e guia de ambiente distinguem resultados executados de CI configurada.

Execuções concluídas em macOS ARM64:

- Python 3.11.17: `make validate` passou com 115 testes base e um skip PyTorch
  antes da instalação do extra; depois de instalar torch 2.10.0, `pytest` aprovou
  119 testes. `make validate-torch` aprovou quatro testes de ponte e treino/recarga
  em processos separados;
- Python 3.12.15: `make validate` passou com 119 testes; `make validate-torch`
  aprovou quatro testes de ponte e treino/recarga em processos separados com torch 2.14.1;
- Ruff/mypy passaram nas versões mínimas e atuais; mypy: 39 fontes em ambos;
- `pip check` passou nos dois ambientes; caches mypy/Matplotlib separados em `/tmp`;
- 12 blocos Python, sem skips PyTorch, e 330 links locais sem destino ausente;
- regressão: RMSE 18603,65 e previsão 350577,02; logística: 64/80 acertos, 6 FP e 10 FN;
  árvore: profundidade 7, erro teste 0,30; MLP: época 92, acurácia 0,775, F1 0,7 e
  p(80,5)=0,651786 preservados nos ambientes;
- gradient check: aproximadamente 4,68e-11 em NumPy 1.26 e 4,31e-11 em NumPy 2.5,
  dentro da tolerância. JSON: 1009 e 1006 bytes, respectivamente; casas decimais
  serializadas podem variar. Não exigir hash idêntico entre ambientes;
- gráficos imobiliário e MLP treino/validação regenerados e inspecionados;
- alta consequência encaminhou para humano; 150 °C/5 mm/s retornou abstenção com
  probability=null. Jev permaneceu offline, sem credenciais ou chamada real.

Preparação realmente executada (uv 0.13.0 foi ferramenta temporária, não uma
dependência obrigatória do curso):

```bash
.venv/bin/python -m pip install --target /tmp/sos-ml-validation-tools uv
/tmp/sos-ml-validation-tools/bin/uv python install 3.11 3.12 --install-dir /tmp/sos-ml-pythons --no-bin --cache-dir /tmp/sos-ml-uv-cache
/tmp/sos-ml-pythons/cpython-3.11.17-macos-aarch64-none/bin/python3.11 -m venv /tmp/sos-ml-py311-min
/tmp/sos-ml-pythons/cpython-3.12.15-macos-aarch64-none/bin/python3.12 -m venv /tmp/sos-ml-py312
/tmp/sos-ml-validation-tools/bin/uv pip install --python /tmp/sos-ml-py311-min/bin/python --cache-dir /tmp/sos-ml-uv-cache -c tests/constraints-min.txt -e '.[dev]'
/tmp/sos-ml-validation-tools/bin/uv pip install --python /tmp/sos-ml-py311-min/bin/python --cache-dir /tmp/sos-ml-uv-cache -c tests/constraints-min.txt 'torch==2.10.0'
/tmp/sos-ml-validation-tools/bin/uv pip install --python /tmp/sos-ml-py312/bin/python --cache-dir /tmp/sos-ml-uv-cache -e '.[dev]'
/tmp/sos-ml-validation-tools/bin/uv pip install --python /tmp/sos-ml-py312/bin/python --cache-dir /tmp/sos-ml-uv-cache -e '.[dev,torch]'
```

Validações realmente executadas:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-311-mpl XDG_CACHE_HOME=/tmp/sos-ml-311-cache MYPY_CACHE_DIR=/tmp/sos-ml-311-mypy PYTHONUTF8=1 PATH="/tmp/sos-ml-py311-min/bin:$PATH" make validate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-311-mpl XDG_CACHE_HOME=/tmp/sos-ml-311-cache PYTHONUTF8=1 /tmp/sos-ml-py311-min/bin/python -m pytest
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-311-mpl XDG_CACHE_HOME=/tmp/sos-ml-311-cache PATH="/tmp/sos-ml-py311-min/bin:$PATH" make validate-torch
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-311-mpl XDG_CACHE_HOME=/tmp/sos-ml-311-cache /tmp/sos-ml-py311-min/bin/python scripts/visualizar_regressao.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-312-mpl XDG_CACHE_HOME=/tmp/sos-ml-312-cache MYPY_CACHE_DIR=/tmp/sos-ml-312-mypy PYTHONUTF8=1 PATH="/tmp/sos-ml-py312/bin:$PATH" make validate
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-312-mpl XDG_CACHE_HOME=/tmp/sos-ml-312-cache PYTHONUTF8=1 PATH="/tmp/sos-ml-py312/bin:$PATH" make validate-torch
MYPY_CACHE_DIR=/tmp/sos-ml-311-mypy /tmp/sos-ml-py311-min/bin/python -m mypy
MYPY_CACHE_DIR=/tmp/sos-ml-312-mypy /tmp/sos-ml-py312/bin/python -m mypy
/tmp/sos-ml-py311-min/bin/python -m pip check
/tmp/sos-ml-py312/bin/python -m pip check
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/sos-ml-312-mpl XDG_CACHE_HOME=/tmp/sos-ml-312-cache /tmp/sos-ml-py312/bin/python -m sos_ml.local_ai inferir --temperature 150 --vibration 5
git diff --check
```

Downloads inicialmente falharam por DNS no sandbox e foram concluídos após
aprovação. Não se instalou Python globalmente nem se substituiu a `.venv` de
trabalho. Sem falhas nos caminhos finais; o aviso de depreciação do SciPy no
Scikit-learn mínimo permanece explícito. Linux/Windows, mínimos de todas as
transitivas e Jev live permanecem sem execução nesta máquina.
