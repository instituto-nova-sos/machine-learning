# Matemática e representação

[← Índice do módulo](README.md) · [Prática →](pratica.md)

Escalar é um número; vetor é uma sequência ordenada; matriz é uma grade bidimensional; tensor é
um arranjo de dimensão arbitrária. Uma pessoa com `[área, quartos, idade]` vira vetor; muitas
pessoas, uma matriz com linhas-exemplo; pesos, vetor ou matriz; lote de imagens coloridas, tensor
com eixos como lote, altura, largura e canais. A forma torna esses significados verificáveis.

`[1,2]+[3,4]=[4,6]`; `2[1,2]=[2,4]`. O produto escalar
`x·w=Σᵢxᵢwᵢ`: `[2,3]·[4,5]=23`. Geometricamente, ele combina magnitudes e alinhamento; em ML,
é a soma ponderada. Matriz-vetor aplica esse cálculo a cada linha. Produto matricial combina
linhas da esquerda com colunas da direita; as dimensões internas devem coincidir. Transposta
troca linhas por colunas.

Operações elemento a elemento não são produto matricial. *Broadcasting* expande dimensões
compatíveis segundo regras da biblioteca; conveniente, mas pode mascarar forma errada.

Vetorização delega laços a rotinas compiladas e reduz a sobrecarga do interpretador Python. Isso
costuma ser mais rápido para arrays grandes, mas o ganho depende de tamanho, memória e operação.
