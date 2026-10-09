# Inferência local: IA também roda no seu computador

[← PyTorch](../pytorch-na-pratica/README.md)

Pré-requisito: treinar, salvar e recarregar. Objetivo: separar modelo, runtime e
aplicação, medindo recursos. IA não significa necessariamente chamar uma API de chatbot.
A regressão inicial já fazia inferência local; agora usamos uma rede pequena.

```text
entrada → validação → representação numérica → runtime/modelo
        → probabilidade → decisão estruturada → política → exibição/revisão
```

Modelo é função+parâmetros; runtime executa operações (NumPy/PyTorch aqui);
aplicação valida entrada e decide efeitos. Treinar ajusta parâmetros com alvos;
inferir usa parâmetros fixos sem alvos, gradientes ou novo fit.

## Prática obrigatória pequena e local

Na raiz, com ambiente base instalado:

```bash
python -m sos_ml.local_ai treinar --plot artifacts/mlp_treino_validacao.png
python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --measure
```

O primeiro comando treina a MLP NumPy, monitora validação, restaura melhor estado
 e salva arquitetura/pesos/scalers no JSON. O segundo inicia novo processo e faz
somente inferência, seguida de política. Não baixa pesos nem consulta serviço remoto.
A versão PyTorch de treino usa `--backend torch`; a demonstração de runtime PyTorch
está em `python -m sos_ml.torch_demo inferir`. Exportar os mesmos pesos para NumPy
não muda a família do modelo: nossa equivalência numérica é testada.

No gráfico de treino/validação, a linha vertical é a época restaurada. O último
ponto do histórico não é necessariamente o estado salvo. A curva de teste não
é usada para ajustar. Os 80 casos finais são os mesmos já usados na trilha antiga:
essa reutilização didática **não constitui um benchmark independente novo**.

## Medir antes de afirmar

A rede 2→4→1 tem `2×4+4+4×1+1=17` parâmetros. float64 usa 8 bytes por valor:
136 bytes de números de parâmetros. float32 usaria 68. Scalers, JSON, buffers,
intermediários e runtime ocupam memória adicional; tamanho do arquivo não é RAM
 e bytes de pesos não são memória total do processo. JSON privilegia inspeção,
não compactação. O comando imprime tamanho do arquivo ao treinar.

`--measure` aquece o caminho e mede 100 chamadas de uma entrada igual. Usa relógio
monotônico `perf_counter`, informa média em ms incluindo escala/modelo/política,
sem carga do arquivo, inicialização, disco, tela ou rede. Meça repetições e quantis
em investigação; uma média não caracteriza latência de cauda. Entradas fora do
domínio abstêm antes do modelo: não compare essa medição com inferência normal.
Latência é tempo por requisição; vazão (throughput) é entradas por unidade de tempo.
Lotes podem melhorar vazão sem reduzir espera de uma entrada individual.

## Hardware, tamanho e precisão

CPU é processador geral, suficiente aqui. GPU favorece operações paralelas grandes,
mas transferência e inicialização podem dominar tarefas minúsculas. NPU é acelerador
especializado; requer operadores/runtime compatíveis. Suporte depende do dispositivo.
Mais parâmetros aumentam armazenamento e operações; tamanho sozinho não mede qualidade.

Quantização representa valores em formatos de menor precisão, com escalas/pontos
zero quando apropriado. Pode reduzir memória/custo, mas introduz erro e exige nova
avaliação das probabilidades **e da política**. Um pequeno arredondamento perto de
limiar pode mudar ação. Nem todo operador/dispositivo ganha velocidade. Não aplicamos
quantização obrigatória a 17 parâmetros: estudamos trade-offs antes de otimizar.

ExecuTorch é uma opção do ecossistema PyTorch para implantação em dispositivos;
exportação, operadores e backend exigem trabalho adicional. É leitura opcional,
sem instalação pesada nesta prática. Nenhum modelo de linguagem grande é necessário.

## Local versus remoto

Local pode oferecer operação offline, privacidade de entradas, menor dependência
de rede e custo previsível por execução. Ainda exige proteção de logs/arquivos,
atualizações, energia, memória, empacotamento e validação de hardware. Offline não
significa que o download/instalação inicial não precise de internet.
Remoto permite delegar recursos e atualizar centralmente, mas adiciona disponibilidade,
latência de rede, custos e envio de dados. Escolha depende de tarefa e contexto.
Probabilístico não significa software sem regras: contratos, política e autorização
continuam determinísticos. Dados sintéticos não autorizam manutenção real.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: decisões](../modelos-de-decisao/README.md)
