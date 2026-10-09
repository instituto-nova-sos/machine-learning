# Soluções do instrutor

Nível 1: dJ/dw=0,924, dJ/db=0,462. Taxa 0,1 atravessa uma região e aumenta J;
0,01 produz w=0,49076, b=0,09538, z=1,0769 e perda menor. Derivar não atualiza.
Nível 2: D₂=[−0,2;0,2]; dW₂=[[0,4],[0,4]], db₂=[0]. Cada exemplo contribui.
O check deve preservar exatamente o estado; exigir tolerância e todos os parâmetros.
Nível 3: não há epsilon universal. Sem 1/n, gradiente da soma cresce com o lote;
usar a mesma taxa deixa de ser o mesmo experimento. ReLU em zero exige análise
separada: a diferença central pode não coincidir com a convenção da implementação.
