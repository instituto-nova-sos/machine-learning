# Exercícios de classificação

[← Prática](pratica.md) · [Índice](README.md) · [Referências →](referencias.md)

## Fundamentos

1. Uma entrada já padronizada é `[1, 0.5]`, os pesos são `[0.8, 0.4]` e o viés é `-0.2`.
   Calcule escore, probabilidade e classes para limiares 0,5 e 0,7. Explique cada resultado.
2. Para a probabilidade anterior, calcule a entropia cruzada quando y=1 e quando y=0.
   Por que as perdas diferem se a previsão é igual?
3. Refaça a primeira atualização de `X=[[-1],[1]], y=[0,1]`, começando em zero com taxa 0,1.
   Calcule as duas probabilidades depois do passo e explique o sinal do gradiente do peso.
4. O modelo produz exatamente 0,5. Qual classe nosso código escolhe com limiar 0,5?
   Essa convenção prova que uma falha ocorrerá?

## Aplicação

1. Execute o projeto completo. Identifique no código onde a partição ocorre e onde cada scaler
   é ajustado. Explique por que usar a média de todos os 400 exemplos seria vazamento.
2. Treine duas instâncias novas com taxas 0,01 e 0,1 por 200 épocas no mesmo treino.
   Compare apenas as perdas de treino. Distingua “aprendeu mais rápido” de “generaliza melhor”.
3. Para a nova medida `[80, 5]`, compare decisões nos limiares 0,3, 0,5 e 0,7 mantendo os pesos.
   O que mudou? Que informação faltaria para escolher um limiar operacional?
4. Explique as formas de `X`, `w`, `p`, `y` e `X.T @ (p-y)` para 320 exemplos e dois atributos.
   Por que `y` com forma `(320,1)` é perigoso ao subtrair um vetor `(320,)`?
5. Uma planilha real contém cem medições da mesma máquina. A divisão aleatória por linha ainda
   é adequada? Proponha uma separação coerente com prever máquinas novas ou períodos futuros.

## Desafio

1. Verifique o gradiente do viés e dos dois pesos por diferenças finitas em um lote pequeno.
   Use epsilon 1e-6 e tolerância, não igualdade exata. Compare Python puro e NumPy.
2. Use `X=[[-1]]*4 + [[1]]*4` e `y=[0,0,0,1,0,1,1,1]`. Antes de treinar, estime a frequência
   de classe 1 em cada valor de x. Depois compare as três implementações e tente deduzir o peso.
3. Explique por que separar perfeitamente dois pontos não é evidência de um modelo industrial
   útil. O que pode acontecer com os pesos sem regularização se você continuar treinando?
4. Redija uma nota de uso: origem sintética, horizonte do alvo, falsos alertas, falhas não
   detectadas, calibração, diferenças entre grupos de máquinas e risco de extrapolação.

[Soluções comentadas do instrutor](solucoes-instrutor/README.md).
