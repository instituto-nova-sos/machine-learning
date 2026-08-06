# Fundamentos de Inteligência Artificial e Machine Learning

[← Índice do módulo](README.md) · [Prática →](pratica.md)

## 1. Um mapa, não uma coleção de palavras da moda

Inteligência Artificial, Machine Learning, Deep Learning e LLM não são sinônimos. Eles também não
ocupam exatamente o mesmo nível conceitual:

```text
Inteligência Artificial
├── abordagens sem Machine Learning
│   ├── regras e representação simbólica
│   ├── busca
│   └── planejamento
└── Machine Learning
    ├── regressão, árvores, métodos de kernel etc.
    └── Deep Learning
        ├── MLPs profundas
        ├── CNNs profundas
        ├── RNNs profundas
        └── Transformers
            └── muitos LLMs modernos
```

Esse mapa expressa a relação `DL ⊂ ML ⊂ IA`. A Figura 1.4 de Goodfellow, Bengio e Courville
apresenta Deep Learning como parte do aprendizado de representações, que é parte de Machine
Learning, usado em muitas — mas não todas — as abordagens de IA.

As fronteiras da IA mudaram historicamente e variam entre autores. A taxonomia é uma ferramenta
para raciocinar, não uma lei natural. Dizer em qual definição estamos trabalhando é mais rigoroso
do que discutir se uma tecnologia “é IA” sem declarar critérios.

## 2. Inteligência Artificial

IA é o campo científico e de engenharia que estuda sistemas computacionais capazes de realizar
tarefas associadas a percepção, raciocínio, aprendizagem, decisão, planejamento, comunicação ou
ação inteligente.

Essa definição não afirma que a máquina:

- possui consciência;
- compreende como uma pessoa;
- é inteligente de forma geral;
- produz respostas verdadeiras;
- age de maneira neutra.

Um sistema pode resolver uma tarefa por busca ou por regras explicitamente programadas. Nesse
caso, ele pode pertencer à IA sem ter aprendido seus critérios a partir de dados.

### Exemplo de regra explícita

```python
def classificar_temperatura(valor: float) -> str:
    if valor >= 38.0:
        return "alerta"
    return "sem alerta"
```

O limite `38.0` foi escrito por uma pessoa. Executar essa regra não constitui treinamento. A regra
pode ser útil e auditável, mas sua adequação ainda depende do domínio e do contexto.

## 3. Machine Learning

Machine Learning é um subcampo da IA no qual algoritmos ajustam modelos usando dados e um
critério de desempenho. Uma formulação clássica de Mitchell descreve aprendizagem em termos de
três elementos:

- **tarefa (`T`)**: o que o sistema deve realizar;
- **experiência (`E`)**: os dados ou interações disponíveis;
- **medida de desempenho (`P`)**: como a melhora será avaliada.

Exemplo:

```text
T: estimar o consumo de energia da próxima hora
E: medições históricas de consumo, horário e temperatura
P: erro absoluto médio em períodos futuros preservados
```

O sistema aprende somente no sentido operacional definido por `T`, `E` e `P`. Melhorar uma
métrica não implica adquirir compreensão geral sobre energia, causalidade ou comportamento humano.

### Programação explícita e ML podem coexistir

A oposição “programação tradicional versus ML” é didática, não absoluta. Um sistema real combina:

- regras escritas por pessoas;
- transformações determinísticas;
- parâmetros ajustados por dados;
- validações, limites e políticas de uso;
- software convencional para servir e monitorar o modelo.

O algoritmo de treinamento, a função de perda e a arquitetura continuam sendo programados. O que
muda é que alguns valores que determinam as previsões são estimados a partir dos dados.

## 4. Deep Learning e aprendizado de representações

Deep Learning é uma abordagem de ML baseada na composição de múltiplas camadas. Cada camada
transforma a representação recebida e fornece outra representação à próxima:

```text
entrada → representação 1 → representação 2 → ... → saída
```

Em uma unidade simplificada:

```text
z = w · x + b
a = g(z)
```

`x` é a entrada, `w` e `b` são parâmetros e `g` é uma função de ativação. Sem ativações não
lineares, compor várias transformações lineares continuaria equivalente a uma única transformação
linear.

“Profunda” se refere à composição de transformações. Não há um número universal de camadas que
separe toda rede rasa de toda rede profunda. Mais camadas aumentam capacidade e podem aprender
representações úteis, mas também aumentam custo, demanda de dados e dificuldade de otimização.

## 5. Arquiteturas de redes profundas

Uma **arquitetura** descreve como operações, parâmetros e fluxos de informação são organizados.
Ela não é a mesma coisa que a tarefa realizada.

### MLP — perceptron multicamada

Uma MLP é uma rede *feedforward*: na inferência, a informação percorre camadas da entrada para a
saída. Em uma camada densa, cada unidade recebe normalmente todas as saídas da camada anterior.

```text
[atributos] → [camada densa + ativação] → [camada densa] → [saída]
```

MLPs são uma base importante para compreender forward propagation, ativações e backpropagation.
Por padrão, porém, não incorporam uma hipótese específica sobre vizinhança espacial ou ordem
temporal.

### CNN — rede neural convolucional

CNNs são especializadas em dados com topologia de grade conhecida, como imagens 2D ou séries 1D.
Pequenos filtros são aplicados em diferentes regiões, introduzindo:

- conectividade local;
- compartilhamento dos mesmos parâmetros entre posições.

Isso expressa a hipótese de que um padrão local pode ser relevante em lugares diferentes e reduz
o número de parâmetros em comparação com uma conexão densa geral.

### RNN — rede neural recorrente

RNNs processam sequências reutilizando parâmetros e mantendo um estado entre posições:

```text
entrada atual + estado anterior → novo estado → saída
```

O mecanismo representa dependência de ordem e admite sequências de comprimentos variados. RNNs
podem sofrer com gradientes que desaparecem ou explodem; LSTM e GRU introduzem mecanismos de
controle do estado para lidar melhor com algumas dependências longas.

### Transformer

Transformers usam atenção para construir representações contextuais. Na self-attention, cada
posição pondera informações de outras posições. Como atenção isolada não codifica automaticamente
a ordem, o modelo precisa receber informação posicional ou mecanismo equivalente.

Um bloco Transformer não é apenas atenção: costuma combinar atenção multi-head, projeções
lineares, MLPs, ativações, normalizações e conexões residuais. Essa arquitetura tornou-se central
em linguagem e também é aplicada a visão, áudio e dados multimodais.

| Arquitetura | Hipótese estrutural | Operação característica |
|---|---|---|
| MLP | atributos formam um vetor | camadas densas compostas |
| CNN | padrões locais podem se repetir numa grade | filtros compartilhados |
| RNN | ordem e estado anterior importam | recorrência com parâmetros compartilhados |
| Transformer | relações entre posições dependem do contexto | atenção e posição |

## 6. Modelos de linguagem e LLMs

Um **modelo de linguagem** atribui probabilidades a sequências ou ao próximo elemento de uma
sequência. Modelos de linguagem são anteriores ao Deep Learning: modelos de n-gramas, por exemplo,
usavam contagens e hipóteses estatísticas locais.

Um **LLM moderno** é um modelo de linguagem de grande escala implementado com Deep Learning,
normalmente um Transformer. Durante um objetivo autoregressivo comum, recebe tokens anteriores e
estima uma distribuição para o próximo token. Repetir a amostragem ou escolha de tokens permite
gerar texto.

Isso esclarece quatro limites:

1. prever tokens não equivale, por definição, a verificar fatos;
2. texto fluente não demonstra consciência ou compreensão humana;
3. o modelo pode reproduzir erros e vieses presentes nos dados e no processo de desenvolvimento;
4. “large” descreve escala, sem um único limiar universal que transforme um modelo em LLM.

## 7. Três eixos que não devemos misturar

Uma descrição completa separa pelo menos três perguntas.

### Eixo A — Como ocorre a aprendizagem?

- **supervisionada:** exemplos possuem alvos usados no ajuste;
- **não supervisionada:** busca-se estrutura sem um alvo supervisionado fornecido;
- **auto-supervisionada:** alvos de treinamento são construídos a partir dos próprios dados;
- **por reforço:** um agente interage e recebe sinais de recompensa.

### Eixo B — Qual é a tarefa?

- **regressão:** prever valores contínuos;
- **classificação:** estimar classes ou probabilidades de classes;
- **agrupamento:** organizar exemplos por algum critério de semelhança;
- **geração:** produzir novas amostras segundo uma distribuição modelada;
- **redução de dimensionalidade:** representar dados com menos dimensões.

### Eixo C — Qual arquitetura ou família de modelo é usada?

- regressão linear ou logística;
- árvore ou conjunto de árvores;
- máquina de vetores de suporte;
- MLP, CNN, RNN ou Transformer.

Esses eixos se combinam. Um Transformer pode ser treinado de forma auto-supervisionada para
geração e depois adaptado de forma supervisionada para classificação. Uma CNN pode classificar
imagens ou integrar um modelo generativo. “Geração” não é, por si só, um paradigma de aprendizagem.

## 8. Vocabulário mínimo

| Termo | Significado neste curso |
|---|---|
| Exemplo | uma unidade de observação, como uma medição horária |
| Atributo | variável de entrada disponível para o modelo |
| Alvo | variável que se deseja aproximar em aprendizagem supervisionada |
| Dataset | coleção estruturada de exemplos e seu contexto |
| Modelo | função ou sistema parametrizado que produz uma saída |
| Algoritmo de treino | procedimento que ajusta parâmetros |
| Parâmetro | valor ajustado durante o treinamento, como peso ou viés |
| Hiperparâmetro | escolha externa ao ajuste, como taxa de aprendizado |
| Função de perda | quantidade otimizada durante o treinamento |
| Métrica | medida usada para avaliar algum aspecto do desempenho |
| Treino | processo de ajuste com os dados de treinamento |
| Inferência | uso do modelo ajustado para produzir saídas |

Perda e métrica podem coincidir, mas não precisam. Podemos otimizar uma função diferenciável e
comunicar outra medida mais interpretável. Também podemos precisar de várias métricas para revelar
tipos diferentes de erro.

## 9. Treinamento e inferência

No treinamento supervisionado:

```mermaid
flowchart LR
    D[Entradas e alvos de treino] --> A[Algoritmo de treinamento]
    O[Objetivo e hiperparâmetros] --> A
    A --> M[Modelo com parâmetros ajustados]
```

Na inferência:

```mermaid
flowchart LR
    X[Nova entrada] --> M[Modelo ajustado]
    M --> Y[Previsão]
```

O alvo real da nova entrada não é fornecido ao modelo no momento da previsão. Quando esse valor se
torna disponível posteriormente, ele pode apoiar avaliação ou monitoramento, respeitando o desenho
experimental e o contexto temporal.

## 10. Antes de escolher ML

Uma formulação responsável começa por perguntas anteriores ao algoritmo:

1. Qual decisão ou tarefa precisa de apoio?
2. Qual é a unidade de análise e o horizonte da previsão?
3. Que informação estará realmente disponível no momento de uso?
4. Existe um alvo observável e adequado, ou apenas um proxy conveniente?
5. Uma regra simples ou melhoria de processo resolveria o problema?
6. Há exemplos suficientes e representativos?
7. Quais são os custos de falsos positivos, falsos negativos e abstenções?
8. Quem pode ser prejudicado, contestar ou corrigir a saída?
9. Como detectar mudança de contexto após a implantação?

ML é inadequado quando o problema não está definido, os dados não sustentam o objetivo, o erro é
inaceitável, a decisão exige garantias que o modelo não oferece ou uma solução determinística mais
simples é suficiente.

## 11. O que uma previsão não garante

Uma previsão é condicionada aos dados, à representação, ao modelo, ao objetivo e ao contexto de
uso. Mesmo uma boa avaliação não prova automaticamente:

- causalidade;
- verdade em cada caso individual;
- desempenho futuro sob mudança de distribuição;
- ausência de discriminação;
- adequação para automatizar uma decisão;
- compreensão humana do conteúdo.

> **Síntese:** compreender IA começa por identificar o que o sistema calcula, de onde vieram os
> dados, o que foi ajustado e qual evidência sustenta seu uso — não por atribuir capacidades humanas
> ao software.

Continue com a [prática de especificação do problema](pratica.md) e consulte as
[referências verificáveis](referencias.md).
