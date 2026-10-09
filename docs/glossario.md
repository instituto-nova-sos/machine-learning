# Glossário

[← Voltar à aula principal](../README.md) · [Referências gerais →](referencias-gerais.md)

- **Algoritmo:** procedimento finito e não ambíguo para realizar uma tarefa.
- **Inteligência artificial (AI):** campo que estuda sistemas computacionais capazes de realizar tarefas associadas a capacidades inteligentes.
- **Aprendizado de máquina (machine learning, ML):** métodos que ajustam comportamento a partir de dados segundo um objetivo.
- **Aprendizado profundo (deep learning):** aprendizado com redes neurais de múltiplas camadas de representação.
- **Modelo:** função ou sistema parametrizado usado para produzir previsões.
- **Parâmetro:** valor ajustado pelo treinamento, como peso ou viés.
- **Hiperparâmetro:** escolha externa ao ajuste, como taxa de aprendizado.
- **Atributo (feature):** variável de entrada representada numericamente.
- **Rótulo ou alvo (label/target):** valor que se deseja prever em aprendizagem supervisionada.
- **Exemplo:** uma observação individual.
- **Conjunto de dados (dataset):** coleção estruturada de exemplos.
- **Treino:** processo de ajustar parâmetros com dados de treinamento.
- **Inferência:** uso do modelo ajustado para produzir uma saída.
- **Validação:** avaliação usada para escolhas durante o desenvolvimento.
- **Teste:** avaliação final em dados preservados das escolhas de desenvolvimento.
- **Perda (loss):** medida numérica do erro de uma previsão; custo costuma agregar perdas.
- **Gradiente (gradient):** vetor de derivadas parciais de uma função escalar.
- **Época (epoch):** passagem completa pelos exemplos de treino.
- **Lote (batch):** subconjunto processado numa atualização.
- **Tensor:** arranjo multidimensional de números.
- **Sobreajuste (overfitting):** adaptação que reduz erro de treino sem generalizar adequadamente.
- **Generalização:** desempenho em exemplos não usados para ajustar o modelo.

- **Avaliação de modelos:** Contagens manuais, matriz, acurácia, precisão, revocação, especificidade, F1 e baseline com métricas indefinidas explícitas; APIs anteriores preservadas.

- **Generalização e sobreajuste:** Partição manual 60/20/20; capacidade, regularização, validação cruzada e experimento deliberado com árvore no mesmo domínio sintético.

- **Neurônio artificial:** Soma ponderada e ativações linear, sigmoid e ReLU em Python puro; equivalência explícita com logística e composição de camadas afins.

- **Redes neurais:** Camada densa e MLP NumPy 2→4→1; formas explícitas e propagação para frente com saída em logits.

- **Retropropagação:** Cadeia escalar em Python puro, gradientes vetorizados da MLP, atualização separada e checagem de todos os parâmetros por diferenças centrais.

- **Deep Learning:** Aprendizado de representações, profundidade, CNN/RNN/Transformer e distinção arquitetura/tarefa, com cálculos pequenos locais.

- **PyTorch na prática:** Tensores CPU, autograd, Module/Parameter/Linear, BCE, SGD, modos, state_dict, persistência e inferência em outro processo; extra torch preservado.

- **Inferência local:** Persistência JSON com arquitetura, parâmetros e scalers; processo separado, CPU, medição de bytes/latência e discussão de quantização/ExecuTorch.

- **Modelos de decisão e revisão humana:** Fronteira local tipada, espaço limitado, política determinística, concentração binária explicitamente definida, abstenção e comparação classificador/decisão/LLM.

- **Jev como estudo de caso:** Choice/Score/Noul verificados na documentação oficial; camada local obrigatória e script remoto opcional separado, sem keys ou chamadas obrigatórias.

## Vocabulário da continuação

- **Matriz de confusão:** contagens entre alvo observado e classe prevista; neste curso,
  alvo nas linhas, previsão nas colunas, classes na ordem 0/1.
- **Precisão (precision):** fração de positivos observados entre alertas: VP/(VP+FP).
- **Revocação/sensibilidade (recall):** fração de positivos detectados: VP/(VP+FN).
- **Especificidade:** fração de negativos reconhecidos: VN/(VN+FP).
- **F1:** 2VP/(2VP+FP+FN); resume precisão e revocação, sem contar VN.
- **Limiar (threshold):** valor externo ao modelo que converte probabilidade em decisão;
  não muda a estimativa nem é garantia de consequência.
- **Regularização:** penalidade/restrição ao ajuste para controlar capacidade; depende
  do problema e é escolhida usando desenvolvimento, sem selecionar pelo teste.
- **Validação cruzada (cross-validation):** treino e avaliação em dobras do desenvolvimento;
  transformações são reajustadas em cada treino de dobra, com teste final preservado.
- **Mudança de distribuição (distribution shift):** diferença entre os dados de
  desenvolvimento e os do uso posterior, inclusive dentro das mesmas faixas numéricas.
- **Propagação para frente (forward propagation):** executar operações da entrada à saída.
- **Retropropagação (backpropagation):** calcular derivadas em sentido inverso usando regra
  da cadeia; a atualização dos parâmetros ocorre depois, pelo otimizador.
- **Grafo computacional:** representação das dependências entre entradas e operações.
- **Autograd:** mecanismo que registra operações compatíveis e calcula/acumula suas derivadas.
- **Logit:** escore real antes da sigmoid, não probabilidade; usado pela BCE estável.
- **Dtype:** tipo de representação numérica, como float32/float64; afeta memória e precisão.
- **Device:** dispositivo de execução/armazenamento do tensor; CPU é o padrão obrigatório aqui.
- **Checkpoint:** estado para retomar treino, incluindo parâmetros e estado do otimizador;
  difere de artefato mínimo de inferência.
- **State_dict:** mapa PyTorch de nomes para parâmetros e buffers; não contém sozinho
  a arquitetura ou o contrato dos sensores.
- **Latência:** tempo por chamada com escopo definido; carga e pipeline aquecido são medições distintas.
- **Vazão (throughput):** quantidade processada por unidade de tempo; não é o inverso universal
  da latência quando há lotes, paralelismo ou espera.
- **Quantização:** representar valores com menor precisão e escalas apropriadas; exige
  reavaliar erros e política, não garante aceleração em qualquer runtime.
- **Abstenção (abstention):** recusar produzir/automatizar uma decisão e encaminhar revisão.
- **Revisão humana (human-in-the-loop):** participação com contexto, responsabilidade e
  possibilidade de recusar, especialmente para consequência elevada ou incerteza.
- **Política da aplicação:** regras determinísticas que autorizam, limitam ou recusam o
  uso de uma saída probabilística; não é sinônimo do modelo aprendido.
- **Confiança (confidence):** estatística cuja semântica depende do provedor. No adaptador
  local é |2p−1|; no Jev siga a primitiva/documentação. Não é garantia de correção.

- **Engenharia de ML:** Contratos versionados, pré-processamento persistido, observabilidade JSONL, reprodutibilidade, segurança de artefatos e responsabilidade humana.

- **Projeto integrador:** Fluxo local completo de sensores sintéticos até modelo, artefato, inferência, decisão, política, revisão e registro; rubrica de 100 pontos e soluções separadas.
