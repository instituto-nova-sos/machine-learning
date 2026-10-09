# Engenharia de ML: o contrato continua depois do treino

[← Jev](../jev-na-pratica/README.md) · [Projeto final](../projeto-integrador/README.md)

Pré-requisito: carregar modelo e aplicar política. Objetivo: reconhecer que dados,
contratos, persistência, observabilidade e autorização pertencem ao sistema.

## Contratos executáveis

Entrada: temperatura em °C e vibração em mm/s, nessa ordem, números finitos.
Representação: cada coluna padronizada pela média/desvio do treino.
Modelo: MLP 2→4→1, float64, logits e BCE no treino, probabilidades na inferência.
Artefato: versão 1, colunas, arquitetura fixa, scalers positivos e arrays finitos.
Saída: ação conhecida, estimativa ou abstenção, concentração e política com motivo.

Contrato de forma não identifica colunas trocadas: nomes/unidades também precisam
ser versionados. O JSON valida ordem declarada, mas a aplicação ainda deve montar
entradas na ordem correta. Hash ajuda a detectar alteração; assinatura/autorização
são necessárias para autenticar origem em produção. Não carregue um pickle para
“descobrir” qual modelo ele contém. Nosso .pt usa weights_only e arquitetura fixa.

## Reprodutibilidade e configuração

Sementes 42/17/23 fixam geração/partição/inicialização; não são evidência de qualidade.
Registre Python/dependências, versão dos dados/artefato e escolhas antes do teste.
Treinar NumPy ou PyTorch usa a mesma inicialização, taxa 0,1 e política de parada.
Não altere hiperparâmetros depois de olhar o teste. A configuração está explícita
nos módulos e args da CLI; um experimento novo deve registrar o que mudou.

A série sintética tem observações independentes. Aplicações reais devem versionar
partições por grupo/tempo, verificar disponibilidade dos atributos e definir quem
pode alterar políticas. Escalar no teste durante inferência seria mudar o contrato.

## Observabilidade e privacidade

```bash
python -m sos_ml.local_ai inferir --temperature 80 --vibration 5 --high-consequence --log artifacts/decisoes.jsonl
```

JSONL é uma linha JSON por evento. O registro inclui versão, marca sintética,
decisão e política/motivo, sem sensores brutos ou segredo. Esse mínimo permite
verificar por que houve revisão; não é trilha completa de produção. Probabilidades
podem revelar contexto; definir acesso/retenção continua necessário. Não registrar
uma chave é parte do contrato, não apenas “cuidado do usuário”.

A CLI não implementa banco, serviço web, alerta real ou confirmação assíncrona.
O resultado `requires_human` é um encaminhamento didático. Uma aplicação real
precisaria fila, responsável, contexto, prazo e registro da decisão humana antes
de qualquer ação consequencial. Separar side effects permite testá-los com mocks.

## Monitorar além da perda

Acompanhe entradas inválidas, fora de domínio, abstenções, taxa de revisão,
latência e versão do artefato. Quando alvos posteriores existirem, avalie erros,
calibração e grupos, respeitando privacidade. Distribuição diferente dentro das
mesmas faixas ainda pode comprometer o modelo. O teste de faixa é proteção mínima,
não detector completo de novidade. Consequência alta continua pedindo confirmação.

Dependências opcionais não devem entrar em import da base. Testes remotos não
participam de `make validate`. `make validate-torch` exige instalação PyTorch e
executa sua ponte; Jev oficial permanece opcional. Registre falhas reais em AGENTS.

[Exercícios](exercicios.md) · [Referências](referencias.md)
