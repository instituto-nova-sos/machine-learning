# Da pergunta à classe

[← Índice](README.md) · [Matemática →](matematica.md)

## 1. Defina o que significa a resposta

Na regressão imobiliária, a resposta era uma quantidade em reais. Aqui a resposta é uma categoria:

| Elemento | Significado neste experimento |
|---|---|
| `x[0]` | temperatura medida agora, em °C |
| `x[1]` | vibração medida agora, em mm/s |
| `y = 0` | não ocorreu falha nas 24 horas seguintes |
| `y = 1` | ocorreu falha nas 24 horas seguintes |
| `p` | probabilidade de classe 1 estimada pelo modelo |
| classe prevista | decisão obtida comparando `p` a um limiar |

O número 1 é um código escolhido para a classe positiva. A classe 0 não significa “equipamento
perfeito para sempre”: o horizonte de 24 horas faz parte da definição. Num projeto real, seria
preciso definir também o que conta como falha, como ela é registrada e quais medições existem
no momento da previsão. Incluir a temperatura medida *depois* da falha seria vazamento.

## 2. Comece com um classificador por regra

Antes de aprender pesos, conseguimos escrever uma regra simplificada:

```python
temperatura = 80.0  # °C
classe = int(temperatura >= 75.0)
print(classe)  # 1: a regra emite um alerta
```

O limiar de 75 °C foi inventado para o exercício. A regra não aprendeu com exemplos e não estima
uma probabilidade. Ela é útil para compreender o que é uma decisão binária. Uma regra pode ser
um baseline, mas seu valor real depende do domínio e de avaliação apropriada.

## 3. Substitua a regra fixa por um escore aprendido

Após padronizar os atributos, a regressão logística calcula:

```text
z = w_temperatura × x_temperatura + w_vibracao × x_vibracao + b
p = sigmoid(z) = 1 / (1 + exp(-z))
classe = 1 se p >= limiar; caso contrário, 0
```

`z` é um escore real: pode ser negativo, zero ou positivo. A sigmoid transforma esse escore em
um valor entre 0 e 1. Valores úteis para conferir à mão: sigmoid(-2) ≈ 0,1192;
sigmoid(0) = 0,5; sigmoid(2) ≈ 0,8808. A exponencial `exp(a)` significa `e` elevado a `a`.

Apesar do nome, regressão logística é usada aqui para **classificação**. Sua saída intermediária
é contínua, mas a tarefa definida pelos rótulos é binária. Os pesos e o viés são aprendidos;
a taxa, o número de épocas e o limiar são escolhas externas ao ajuste dos pesos.

## 4. Probabilidade e decisão respondem a perguntas diferentes

Se o modelo devolve `p = 0,60`, ele estima 60% de probabilidade de falha sob suas hipóteses.
Isso não garante o desfecho daquele equipamento. Também não garante que, entre cem casos
semelhantes, exatamente sessenta falharão. Verificar a correspondência entre probabilidades
estimadas e frequências observadas é estudar **calibração**, que não é garantida pela sigmoid.

| Probabilidade fixa | Limiar | Classe prevista |
|---|---|---|
| 0,60 | 0,30 | 1 |
| 0,60 | 0,50 | 1 |
| 0,60 | 0,70 | 0 |

Diminuir o limiar amplia o conjunto de alertas sobre as mesmas probabilidades. Pode capturar
mais falhas e também gerar mais falsos alertas. A probabilidade não muda. Neste projeto,
o empate pertence à classe 1. Use nosso comparador explícito para manter essa convenção nas
três versões; não dependa da convenção de empate de `predict()` de outra biblioteca.

O limiar 0,5 é didático. Uma escolha operacional exigiria custos de erros, capacidade da equipe,
validação separada e conhecimento do domínio. **Não escolha o limiar observando o teste.**

## 5. A fronteira continua linear

Para limiar 0,5, a fronteira é o conjunto de entradas com `z=0`, porque sigmoid(0)=0,5.
Com dois atributos, ela é uma reta. A sigmoid curva a relação entre escore e probabilidade,
mas não transforma essa fronteira numa curva arbitrária.

Para um limiar `t` estritamente entre 0 e 1, a fronteira satisfaz:

```text
w₁x₁ + w₂x₂ + b = log(t / (1-t))
```

Mudar `t` desloca a fronteira sem treinar novamente. Com pesos não nulos, as retas são paralelas.
Se todos os pesos forem zero, não existe a fronteira usual: a probabilidade é constante.
Os limiares extremos 0 e 1 exigem tratamento à parte, pois o logaritmo acima não é finito.

## 6. Um acerto não conta toda a história

Um falso alerta ocorre quando prevemos 1 e o alvo é 0. Uma falha não detectada ocorre quando
prevemos 0 e o alvo é 1. Esses erros podem ter consequências distintas. Contar acertos e comparar
com “sempre prever a classe mais frequente no treino” inicia a análise, mas não a encerra.

O mundo sintético foi gerado com uma relação logística: o modelo recebe um problema favorável à
sua própria família matemática. Isso permite verificar o algoritmo, mas não demonstra desempenho
industrial. Máquinas de fabricantes, idades e ambientes diferentes podem ter erros diferentes;
sensores podem falhar; temperatura pode refletir carga, ambiente e manutenção. Coeficientes não
provam causas. Um sistema real precisaria de avaliação contextual e supervisão competente.
