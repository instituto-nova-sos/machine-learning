# Soluções comentadas — acesso do instrutor

[← Exercícios](../exercicios.md) · [Índice](../README.md)

## Fundamentos

1. `z=0,8`, `p≈0,689974`. Classes 1 em 0,5 e 0 em 0,7. O escore agrega atributos; a
   sigmoid o transforma em estimativa de probabilidade; o limiar produz uma decisão.
2. Para y=1, `-log(p)≈0,371101`; para y=0, `-log(1-p)≈1,171101`. A qualidade da previsão
   depende do desfecho observado; um erro confiante recebe penalidade maior.
3. `dw=-0,5`, `db=0`; novo peso 0,05 e viés zero. Probabilidades ≈0,487503 e ≈0,512497,
   perda ≈0,668460. O gradiente negativo faz o peso aumentar, elevando p em x=1 e reduzindo
   p em x=-1. Há duas observações nesta conta, não uma estimativa de desempenho real.
4. Classe 1 por convenção de empate inclusivo. Uma regra de decisão não garante um evento futuro.

## Aplicação

1. `prepare_equipment_data` obtém índices antes de chamar `StandardScaler1D.fit` em cada
   coluna de treino. A transformação do teste reutiliza esses objetos. Se a escala dependesse
   do teste, o treinamento receberia informação sobre dados reservados à avaliação.
2. Instancie um modelo novo por taxa: `fit` continua do estado atual. Neste conjunto e nesse
   intervalo de treino, a taxa 0,1 tende a reduzir a perda mais rapidamente. Peça resultados
   observados e não aceite a conclusão automática “taxa maior sempre é melhor”. Nenhuma dessas
   curvas estima sozinha generalização. Não use o teste para decidir a taxa.
3. p≈0,5696: classes 1, 1 e 0. Os pesos e a probabilidade permanecem fixos. Custos, capacidade
   de manutenção, criticidade, dados representativos e validação separada faltam para decidir.
4. X=(320,2), w=(2,), p=y=(320,), X.T=(2,320), dw=(2,). Broadcasting de `(320,1)` com
   `(320,)` produz `(320,320)`: mistura resíduos entre exemplos.
5. Para máquinas novas, preserve grupos de equipamento; para futuro, considere corte temporal
   e disponibilidade das informações. Pode ser necessário combinar grupos e tempo conforme o uso.

## Desafio

1. Para cada parâmetro θ, calcule `[J(θ+ε)-J(θ-ε)]/(2ε)` restaurando θ após a comparação.
   Em valores moderados, diferença absoluta abaixo de 1e-8 funciona no teste incluído.
   ε muito pequeno aumenta cancelamento numérico; ε grande aproxima mal a derivada local.
2. Em x=-1, 1/4 dos alvos é positivo; em x=1, 3/4. A solução tem viés 0 e peso
   `log(3)≈1,098612`, pois sigmoid(-log(3))=0,25 e sigmoid(log(3))=0,75. A sobreposição
   de classes evita separação perfeita e fornece um ótimo finito para conferir os modelos.
3. Em dados perfeitamente separáveis, aumentar a magnitude dos pesos pode reduzir a perda
   indefinidamente, sem mínimo finito de parâmetros. Duas observações não representam uma
   população nem medem calibração, robustez, custos ou segurança.
4. Exija reconhecimento de que a relação logística já foi escolhida no gerador. As faixas
   de sensores e a prevalência são fictícias; os coeficientes não estabelecem causalidade;
   novos tipos de máquinas ou sensores podem deslocar a distribuição. Probabilidades e
   decisões exigiriam validação contextual, e não autorizam controle automático de máquinas.
