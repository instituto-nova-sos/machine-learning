# Matemática da otimização

[← Índice do módulo](README.md) · [Prática →](pratica.md)

Uma função objetivo formaliza o que será otimizado. *Loss* pode medir um exemplo; *cost* costuma
agregar exemplos, embora autores variem. No espaço `(w,b)`, cada ponto é um modelo e a altura pode
ser seu MSE. O gradiente reúne inclinações locais. A atualização é
`θ_{t+1}=θ_t-η∇J(θ_t)`, em que `η` é a taxa de aprendizado.

Taxa pequena faz passos lentos; taxa grande pode atravessar o vale, oscilar ou divergir. Época é
uma passagem pelos dados. Gradiente em lote usa todos; estocástico usa um; mini-lote usa um grupo.
Os últimos são estimativas mais ruidosas e baratas por atualização.

Em regressão linear com MSE, a superfície é convexa e o mínimo global é acessível sob condições
adequadas. Isso não se generaliza a todo modelo: objetivos podem ter mínimos locais, selas e
regiões planas.

> **Equívoco comum:** gradiente descendente não garante o melhor resultado em qualquer problema.
> Garantias dependem da função, passo, inicialização, precisão e critério de parada.
