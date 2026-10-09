# Exercícios

## Nível 1 — compreensão

Explique forma/dtype/device e cada linha do laço. Calcule manualmente gradientes
do escalar e compare `.grad`. Por que eval não substitui no_grad?

## Nível 2 — implementação

Copie a MLP NumPy para Linear com transpostas. Compare forward, BCE e cada gradiente.
Use MSELoss para [3,4,5] versus [2,4,6] e confira a conta manual. Salve e recarregue
state_dict sem serializar a classe inteira.

## Nível 3 — investigação

Execute dois backward em grafos novos sem limpar gradientes: explique acumulação.
Compare float32/64 em um lote pequeno. Recarregue em novo processo e verifique que
não há treino. Liste o que faltaria para retomar exatamente um checkpoint e por
que artefatos desconhecidos não devem ser carregados para “ver o que fazem”.

[Soluções do instrutor](solucoes-instrutor/README.md)
