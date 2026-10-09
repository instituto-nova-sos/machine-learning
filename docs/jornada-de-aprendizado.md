# Jornada de aprendizado

[← Índice principal](../README.md) · [Metodologia](metodologia.md)

Os diretórios representam conceitos, não números de aulas. Cada etapa combina
intuição, cálculo pequeno, implementação reutilizável, testes, limites e exercícios
em níveis. O tempo de estudo depende da turma; avance pelo domínio das competências.

| Etapa | Competência para avançar |
|---|---|
| [Fundamentos de IA](../fundamentos-ia/README.md) | distinguir IA/ML/DL, arquitetura e tarefa |
| [Aprendizagem com dados](../aprendizagem-com-dados/README.md) | definir alvo, partições e vazamento |
| [Matemática para ML](../matematica-para-ml/README.md) | funções, somas, derivadas e cadeia |
| [Vetores e matrizes](../vetores-e-matrizes/README.md) | conferir eixos e produto escalar |
| [Regressão linear](../regressao-linear/README.md) | prever, medir MSE e atualizar sem fit de biblioteca |
| [Otimização](../otimizacao-e-gradiente/README.md) | relacionar taxa, gradiente e convergência |
| [Classificação](../classificacao/README.md) | distinguir logit, probabilidade, classe e BCE |
| [Avaliação](../avaliacao-de-modelos/README.md) | contar VP/VN/FP/FN e interpretar métricas/baseline |
| [Generalização](../generalizacao-e-overfitting/README.md) | selecionar pela validação e reconhecer sobreajuste |
| [Neurônio](../neuronio-artificial/README.md) | recompor produto escalar, viés e ativação |
| [Redes](../redes-neurais/README.md) | conferir X/W/b/Z/A e forward da MLP |
| [Retropropagação](../backpropagation/README.md) | derivar, verificar e atualizar em etapas distintas |
| [Deep Learning](../deep-learning/README.md) | relacionar representações e arquiteturas sem magia |
| [PyTorch](../pytorch-na-pratica/README.md) | explicar tensores/autograd/SGD e recarregar state_dict |
| [Inferência local](../inferencia-local/README.md) | usar CPU/artefato sem treino e medir recursos |
| [Decisões](../modelos-de-decisao/README.md) | separar recomendação, política e ação; recusar automação |
| [Jev](../jev-na-pratica/README.md) | reconhecer Choice/Score/Noul e limites com estudo offline |
| [Engenharia](../engenharia-de-ml/README.md) | versionar contrato e observar decisões sem expor segredos |
| [Projeto final](../projeto-integrador/README.md) | integrar fluxo local, revisão e rubrica |

Até classificação há duas famílias familiares: preço sintético e falha sintética.
A continuação usa o mesmo domínio de equipamentos para evitar trocar o problema
 enquanto aprendemos novas operações. Uma rede maior não é automaticamente melhor.

Python puro precede NumPy. Scikit-learn compara implementações quando apropriado;
PyTorch automatiza operações já derivadas. Seu extra é instalado nessa etapa e roda
em CPU. O projeto final oferece o caminho NumPy e a comparação PyTorch; não exige
GPU, conta paga, nuvem ou download de pesos de modelos grandes.

Jev oficial é **opcional**: a camada local obrigatória implementa arquitetura de
decisão estruturada com modelo do próprio curso, identificada como educacional.
Fixture e testes não consultam rede. A rubrica não depende de acesso ao serviço.

Não avance aceitando só “o teste passou”: explique símbolos, formas, unidades,
denominadores e limitações. Soluções do instrutor estão separadas em cada módulo.
O teste não seleciona hiperparâmetros; as métricas conhecidas do curso são
ilustrações, não novos benchmarks independentes ou evidência industrial.

O modelo é um componente. Dados, software, avaliação, política, incerteza,
segurança, observabilidade e responsabilidade humana completam o sistema.

- [Engenharia de ML](../engenharia-de-ml/README.md): Contratos versionados, pré-processamento persistido, observabilidade JSONL, reprodutibilidade, segurança de artefatos e responsabilidade humana.

- [Projeto integrador](../projeto-integrador/README.md): Fluxo local completo de sensores sintéticos até modelo, artefato, inferência, decisão, política, revisão e registro; rubrica de 100 pontos e soluções separadas.
