# Soluções do instrutor

Nível 1: w.grad=0,924, b.grad=0,462. zero_grad limpa acumulação; backward deriva;
step atualiza. eval escolhe comportamento das camadas, no_grad desliga grafo.
Nível 2: W Linear é transposta. Gradientes também voltam por transposta; comparar
com tolerância. MSE=(1+0+1)/3=2/3. Carregar arquitetura conhecida e weights_only=True.
Nível 3: a segunda chamada acumula, não substitui. Reconstruir forward cria novo
grafo. float32 pode divergir nos últimos bits; não prometer equivalência exata.
Checkpoint exige estado do otimizador e controle de aleatoriedade/época/dados.
Inferência só lê artefatos aprovados, escala entradas e calcula forward.
