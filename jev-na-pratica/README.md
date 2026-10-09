# Jev / System One: um estudo contemporâneo de decisões tipadas

[← Decisões locais](../modelos-de-decisao/README.md)

Pré-requisito: modelo ≠ política ≠ ação. Objetivo: reconhecer uma interface
especializada, avaliar seus limites e preservar a execução local do curso.
Consulta às fontes oficiais: **2 de outubro de 2026**. Reconfira versões ao usar.

## O padrão arquitetural

`ESTADO → PERGUNTA LIMITADA → DECISÃO → POLÍTICA DA APLICAÇÃO → EXIBIÇÃO/REVISÃO`.

Segundo a [introdução oficial](https://docs.typesafe.ai/introduction), Jev é o
modelo da TypeSafe apresentado como System One para respostas estruturadas a
perguntas tipadas. Essa é a descrição do provedor, não uma nova categoria que
substitui ML/DL. Não derivamos sua arquitetura interna a partir da API.
Um LLM generativo modela tokens e pode escrever respostas abertas; a interface
Jev avalia estado e perguntas limitadas e devolve valores estruturados.
Isso não demonstra superioridade universal nem dispensa avaliar dados representativos.

## As primitivas verificadas

| Primitiva | Pergunta | Retorno | Como interpretar |
|---|---|---|---|
| [Choice](https://docs.typesafe.ai/primitives/choice) | uma opção entre critérios nomeados | choice, probabilities, confidence | distribuição sobre as opções, não ação autorizada |
| [Score](https://docs.typesafe.ai/primitives/score) | posição numa rubrica ordenada | score, probabilities, confidence e legend | média ponderada dos níveis, não probabilidade de falha |
| [Noul](https://docs.typesafe.ai/primitives/noul) | afirmação sim/não | noul em [0,1] | P(sim); não há confidence separado |

Choice recebe critérios como mapa nome→descrição; inclua fallback quando necessário.
Score recebe lista ordenada; posições começam em zero. Níveis [0,1,2] com
probabilidades [0,1;0,6;0,3] dão score `0×0,1+1×0,6+2×0,3=1,2`.
Noul=0,9 significa estimativa de “sim” à afirmação específica, não intensidade.

A [documentação de confiança](https://docs.typesafe.ai/confidence) descreve
confidence de Choice/Score como estatística da distribuição. Para Choice com n
opções, a fórmula documentada é `(p_max−1/n)/(1−1/n)`; não é simplesmente p_max.
Com três opções e máximo 0,8 resulta 0,7. Não substitua pela concentração binária
do adaptador local. Confiança alta **não garante correção**, calibração no domínio
 ou permissão de executar. Uma opção necessária omitida limita a resposta.

## Camada A — obrigatória e local

Nosso `LocalDecisionModel` usa a MLP treinada aqui, não Jev. Reexecute
`python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --high-consequence`.
O que compartilhamos com o estudo de caso é arquitetura de software: fronteira,
contrato, política e humano. Não é reprodução do treinamento ou do modelo oficial.

```bash
python scripts/jev_opcional.py
python -m pytest tests/test_jev_boundary.py
```

O script sem flags usa **resposta fixa de teste (fixture) inventada**, não importa SDK e não abre conexão.
Ele mostra Choice sintética e revisão pela política. As probabilidades são sobre
opções, nunca trocadas por p(falha) do classificador. Testes recusam tipos/ações,
valores, distribuição e escolhas inválidos sem testar disponibilidade do serviço.

Não adotamos pesos externos: as páginas oficiais consultadas documentam serviço
 e SDK, não estabelecem um pacote de pesos local com licença e requisitos verificados.
A licença MIT do SDK não é licença do modelo. A prática local usa nosso modelo MIT,
17 parâmetros e nenhum download pré-treinado. Uma alternativa futura exigiria
origem, licença dos pesos, tamanho/download, manutenção, Python e medição em CPU;
wrappers de prompts de LLM devem ser identificados como tal.

## Camada B — integração oficial opcional

O [SDK oficial](https://docs.typesafe.ai/sdk/python), ligado ao
[repositório typesafe-ai](https://github.com/typesafe-ai/typesafe-sdk-python),
publica `typesafe-sdk`; o pyproject consultado declara versão 0.7.2, MIT e Python >=3.10.
Fixamos essa versão no extra `jev` para este exercício. Ela não participa da
instalação base, avaliação da turma ou `make validate`.

```bash
python -m pip install -e ".[jev]"
```

Defina `TYPESAFE_API_KEY` no ambiente da sua sessão com uma chave obtida por você.
Não cole chave em exemplos, histórico compartilhado, logs ou commits. Não há `.env`
necessário: seguimos ambiente, sem adicionar loader. Acesso pode requerer conta,
rede e cobrança: esta atividade é opcional. Nenhuma chave foi usada nesta evolução.

Somente quando você escolher usar o serviço:

```bash
python scripts/jev_opcional.py --live
```

Leia o script antes. O SDK usa `TypeSafeClient().system_one(state=...,questions=...)`
com Choice/Score/Noul. O exemplo transmite apenas relato e medidas fictícias,
fixa `jev-1.13.0`, lê respostas tipadas e passa Choice pela validação e política.
Uma decisão de consequência alta continua requerendo humano. O caminho remoto
não foi executado com credencial; não confundimos checagem offline com resultado real.

A [página oficial de modelos](https://docs.typesafe.ai/models) registra a versão
 e aliases: `jev-latest` pode mudar. Entrada é texto/JSON de conteúdo textual;
não trate isso como suporte nativo a imagem/áudio. O provedor alerta diferenças
entre idiomas: avalie exemplos **em português** do seu domínio. Não copiamos
promessas de velocidade, custo ou “zero erros” para conclusões do curso.

## Escolher por problema

Julgamentos limitados podem apoiar roteamento, classificação, rubricas, guardrails
(regras de proteção) e portões de fluxo. Escrita, explicação aberta, brainstorming
 e geração de código continuam situações favoráveis a modelos generativos quando
adequadamente avaliados. Regras determinísticas podem ser suficientes para muitos
contratos. Nenhuma interface tipada corrige automaticamente perguntas ambíguas,
viés, dados ausentes, mudança de distribuição ou política inadequada.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Projeto integrador](../projeto-integrador/README.md)
