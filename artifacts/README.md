# Artefatos

[← Voltar à aula principal](../README.md) · [Regressão linear](../regressao-linear/README.md)

Modelos e gráficos gerados localmente aparecem aqui e são ignorados pelo Git. Um arquivo JSON
é legível para fins didáticos; sistemas reais exigem validação de esquema, integridade, acesso,
versionamento e cuidado com formatos de serialização não confiáveis.

`python -m sos_ml.classify --plot artifacts/classificacao_e_perda.png` produz a curva de
entropia cruzada e as fronteiras de decisão do projeto de equipamentos. Apenas pontos de
treino são desenhados; o gráfico não seleciona limiares pelo teste. O classificador não é
serializado nesta fase: o comando regenera dados e refaz o treino a cada execução.
