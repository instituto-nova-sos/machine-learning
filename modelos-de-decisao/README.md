# Da previsão à decisão: a aplicação define a política

[← Inferência local](../inferencia-local/README.md)

Pré-requisito: probabilidade, limiar e persistência. Objetivo: distinguir saída
aprendida, recomendação estruturada, autorização e efeito externo.

```text
regressão:    x → valor contínuo
classificação: x → p → limiar → classe
decisão: estado → pergunta limitada → recomendação → política → exibição/revisão
```

## Ação não sai automaticamente da probabilidade

p=0,71 estima falha no mundo sintético. Não significa “pare a máquina”. Nossa
abstração local usa três ações conhecidas: `operacao_normal`, `agendar_inspecao`,
`revisao_humana_imediata`. A aplicação exibe status; não liga/desliga equipamentos.
Limiares ilustrativos 0,3/0,7 organizam recomendação, sem validação industrial.

Um espaço limitado facilita validar respostas, mas não inclui opções omitidas.
Nosso contrato exige as três opções, incluindo revisão. Uma interface limitada
não faz um modelo tradicional virar um novo modelo de fundação: este é um
**adaptador educacional de decisões estruturadas**, não Jev.

## Concentração, abstenção e revisão

A concentração local `confidence=|2p−1|` vale zero em p=0,5 e um nos extremos.
É uma estatística da distribuição binária; **não é probabilidade de acerto da ação**.
Não é confiança industrial calibrada nem uma implementação das primitivas oficiais.
Uma saída concentrada pode estar completamente errada em domínio novo.

A política verifica, nesta ordem: faixa/contexto, consequência elevada, ambiguidade
 e sugestão de revisão. Concentração menor que 0,3 solicita humano. Consequência
elevada exige humano mesmo com p extremo. Valores fora de 40–100 °C ou 0,5–8 mm/s
abstêm-se antes da rede; esse teste de faixa não detecta toda mudança de distribuição.
Uma falha conhecida de provedor leva a revisão, nunca operação normal silenciosa.

A revisão humana (human-in-the-loop) precisa ter contexto, motivo, responsabilidade
 e possibilidade de recusar. Uma pessoa apenas clicando automaticamente não resolve
viés ou incerteza. Não apresentamos confirmação aqui como controle industrial.

```python
from sos_ml.decisions import EquipmentState, LocalDecisionModel, apply_policy
from sos_ml.equipment_io import load_equipment_model
from pathlib import Path

provider = LocalDecisionModel(load_equipment_model(Path("artifacts/equipamento_mlp.json")))
state = EquipmentState(80, 5, high_consequence=True)
decision = provider.decide(state, ["operacao_normal", "agendar_inspecao", "revisao_humana_imediata"])
print(decision)
print(apply_policy(state, decision))  # exige humano independentemente da concentração
```

Antes gere o artefato com `python -m sos_ml.local_ai treinar`. Depois execute:

```bash
python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --high-consequence
python -m sos_ml.local_ai inferir --temperature 150 --vibration 5
python -m pytest tests/test_decisions.py
```

Leia [decisions.py](../src/sos_ml/decisions.py). `DecisionModel` é fronteira de
provedor; `apply_policy` é código determinístico; retorno não é side effect.
O JSON permite comparar o que o modelo sugeriu com o que a política autorizou.

## Três sistemas para problemas diferentes

| Sistema | Fluxo típico | Força | Limitação e avaliação |
|---|---|---|---|
| Classificador tradicional | atributos→p→classe | pequeno, tarefa mensurável, local | representação/dados específicos, domínio limitado |
| Decisão estruturada | estado+pergunta limitada→resposta tipada→política | integração e espaço explícito | opções ruins/omitidas, confiança não garante correção |
| LLM generativo | contexto→tokens→texto/código/estrutura | escrita, explicação, criação aberta | validação da saída e custo de geração |

Um classificador com adaptador pode implementar o segundo **padrão de sistema**.
Jev é um exemplo de modelo especializado com interface estruturada; não é sinônimo
do padrão. LLMs também podem produzir JSON validado; formato não demonstra calibração.
Escolha por dados, tarefa, avaliação e custo. Escrever uma explicação aberta pode
favorecer geração; roteamento limitado pode favorecer decisão ou regra simples.
Nenhum paradigma substitui universalmente os demais. A saída probabilística continua
precisando de política e validação, especialmente em ações consequenciais.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: Jev](../jev-na-pratica/README.md)
