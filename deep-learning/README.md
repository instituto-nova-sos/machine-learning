# Deep Learning: aprender representações por composição

[← Retropropagação](../backpropagation/README.md)

Pré-requisito: MLP, perda e gradientes. Aprendizado profundo (Deep Learning) é ML
com redes de sucessivos níveis de representação. Não é sinônimo de chatbot, nuvem
ou geração de texto. Nossa rede pequena serve para inspecionar operações; não há
um número universal de camadas que defina “profunda”.

## O que uma representação aprendida muda?

Na logística, nós fornecemos temperatura e vibração e aprendemos uma combinação
linear. Na MLP, `H=ReLU(XW₁+b₁)` também aprende combinações intermediárias; a saída
opera sobre H. Gradientes de W₁ dependem da tarefa via W₂. A representação não é
uma coluna humana fixa nem garantia de explicabilidade. Mais profundidade aumenta
possibilidades e dificuldades: custo, otimização, sobreajuste e gradientes pequenos.

## Operações conhecidas, organizações diferentes

| Arquitetura | Estrutura | Ligação com nossas contas | Limitação |
|---|---|---|---|
| MLP | vetores | produtos matriciais e ativações densas | não impõe vizinhança ou tempo |
| CNN (rede convolucional) | grades | somas ponderadas locais com pesos compartilhados | hipótese local depende dos dados |
| RNN (rede recorrente) | sequências | estado h_t=g(x_t W_x+h_{t−1} W_h+b) | dependências longas dificultam gradientes |
| Transformer | relações entre posições | projeções e atenção, além de MLPs | memória/custo e representação da ordem |

Um filtro `[1,-1]` aplicado à sequência `[2,5,4]` por correlação cruzada produz
`[2−5,5−4]=[−3,1]`. O mesmo par de pesos reaparece nas posições: compartilhamento.
Bibliotecas de CNN frequentemente chamam essa operação de convolução mesmo sem
inverter o filtro. Não confunda filtro com novo conjunto de pesos por posição.

Em uma RNN escalar h_t=0,5h_{t−1}+x_t, com h₀=0 e x=[2,1], h₁=2 e h₂=2.
O estado transmite parte do passado e os mesmos parâmetros são reutilizados.

A atenção calcula pesos de relação: `softmax(QK.T/sqrt(d_k)) @ V`. Q, K e V são
projeções de representações; softmax normaliza escores em cada linha. Com escores
[0,0], os pesos são [0,5;0,5] e a saída é a média dos dois vetores V. Informação
posicional é necessária para representar ordem; atenção sozinha não a codifica.
Transformer também usa normalização, conexões residuais e camadas densas.

## Arquitetura ≠ tarefa

Classificação, regressão e geração são tarefas. MLP/CNN/RNN/Transformer são
organizações de operações. LLM (modelo de linguagem de grande porte) é uma família
de modelos profundos, frequentemente Transformers, usada para modelar linguagem.
Uma CNN pode gerar imagens; um Transformer pode classificar. Tamanho não escolhe
a solução por nós. Avaliação deve medir tarefa, dados representativos e custos.

## Prática sem modelos grandes

Reexecute `python -m sos_ml.network_demo` e localize H e seu gradiente. Faça à mão
a correlação, recorrência e atenção uniformes acima. Depois use as referências
para reconhecer produtos, compartilhamento e composição em arquiteturas maiores.
Nenhum download de pesos é necessário. Um gerador sintético compatível com a
logística não justifica usar uma rede maior ou prever segurança real.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: PyTorch](../pytorch-na-pratica/README.md)
