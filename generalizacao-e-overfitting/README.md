# Generalização: a perda cai, mas para quem?

[← Avaliação](../avaliacao-de-modelos/README.md) · [Trilha](../docs/jornada-de-aprendizado.md)

Pré-requisito: métricas e baseline. Objetivo: distinguir ajuste de parâmetros de
escolha do modelo e observar sobreajuste (overfitting) em números.

## Três responsabilidades

Treino ajusta pesos. Validação compara escolhas: taxa, capacidade, épocas, limiar.
Teste faz a avaliação final depois de congelar essas escolhas. Nosso divisor
manual usa 400 linhas → 240 treino, 80 validação, 80 teste. Escalas aprendidas devem
usar somente as 240 linhas. Uma divisão por linha é apropriada aqui porque o
gerador cria observações independentes; não para medições repetidas da mesma máquina.

Subajuste (underfitting) ocorre quando o modelo ou treino não captura relações úteis,
com erro elevado inclusive no treino. Sobreajuste ocorre quando detalhes e ruído do
treino são aprendidos sem melhoria equivalente fora dele. Capacidade é a variedade
de funções representáveis. Mais parâmetros ou regiões podem aumentar capacidade,
mas capacidade, qualidade do treino e representatividade dos dados são questões distintas.

A intuição de viés/variância compara erro sistemático de uma família restrita com
sensibilidade às amostras. Não é uma identidade numérica universal para toda métrica.
No mundo Bernoulli, parte do desfecho é aleatório: nem conhecer a probabilidade
geradora permite acertar todos os sorteios.

## Veja o fenômeno

Uma árvore usa perguntas como “temperatura <= limite?” para dividir regiões.
Profundidades maiores fazem subdivisões que podem isolar observações de treino.
É um contraste com a fronteira linear da logística, não uma etapa substituta dela.

```bash
python -m sos_ml.generalization --plot artifacts/generalizacao.png
python -m pytest tests/test_generalization.py
```

O comando compara profundidades 1 a 12 em treino e validação, escolhe o menor erro
de validação (menor profundidade em empate) e só então avalia essa árvore no teste.
No gráfico, siga a curva de treino caindo e a de validação deixando de melhorar.
A curva de teste não é usada para selecionar. Mesmo a validação pode ser sobreajustada
se fizermos tentativas demais; registre tentativas e preserve um teste realmente final.

## Limitar o ajuste

Regularização adiciona restrições ou penalidades. Em `J_regularizado = J + λΣw²`,
L2 penaliza magnitudes e acrescenta `2λw` ao gradiente. L1 usa `λΣ|w|`, pode produzir
esparsidade e não tem derivada única em zero. Penalizar pesos depende da escala;
por convenção frequentemente não penalizamos o viés. λ é escolhido na validação.
Limitar profundidade também restringe capacidade, sem ser uma penalidade L1/L2.

Parada antecipada (early stopping) monitora validação, salva o melhor estado e
interrompe após um número de épocas sem melhora. É preciso **restaurar** o melhor
estado; interromper não torna automaticamente os últimos pesos os melhores.

Validação cruzada (cross-validation) divide o desenvolvimento em k partes: treina
em k−1, valida na restante e repete. Reajuste scaler **em cada treino de cada dobra**.
Uma `Pipeline` do sklearn automatiza essa ordem. O teste final continua separado.
Dobra temporal respeita o passado; divisão por grupo preserva identidades de máquinas.

## Experimentos enganosos e limites

Acurácia de treino perfeita pode ser memorização. Coluna criada depois da falha,
normalização antes da divisão ou duplicatas nos dois conjuntos causam vazamento
(data leakage). Adicionar o próprio rótulo a X produz números excelentes e um
sistema inútil: essa entrada não existe no momento de prever.

Mudança de distribuição (distribution shift) ocorre quando novos dados diferem
em sensores, contexto ou prevalência. Um domínio sintético simples não valida outro
domínio. Medir erros por grupos pode revelar proxies e exclusões; não infira causas
pelos pesos. Não tente “corrigir” um teste ruim retreinando até ele ficar bom.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: neurônio](../neuronio-artificial/README.md)
