# Exercícios em três níveis

[← Guia](README.md) · [Aula 1](construcao.md) · [Aula 2](personalizacao.md)

Os níveis organizam dificuldades dentro das duas aulas. O nível 3 é uma escolha
para duplas que terminarem antes, não uma terceira aula ou requisito adicional.

## Nível 1 — compreensão (aula 1)

1. Para `Pagar boleto hoje; estudar Python`, escreva as duas tarefas, categorias e
   prioridades que as regras produzem. O que muda se apagar `hoje`?
2. Explique por que o arquivo só aparece após confirmar. Onde ficam ID e status?
3. Desenhe os vetores de contagens de `boleto estudar` e `estudar estudar` usando
   o vocabulário `[boleto, estudar]`. O que se perde nessa representação?
4. Por que `não é urgente` recebe alta? Quem corrige? Essa regra foi aprendida?
5. Uma resposta JSON com categoria `saúde` deve ser salva? E uma tarefa inventada
   com campos válidos? Diferencie validação estrutural de revisão de significado.

## Nível 2 — implementação (aula 2)

Escolha um dos seis contextos do guia. Acrescente uma palavra à regra existente,
registre antes/depois e escreva um teste para a mudança. Modelo de teste para
quem escolheu acrescentar `redação` a estudos (salve como `tests/test_minha_rotina.py`):

```text
from sos_ml.rotina.rules import category_by_rules


def test_redacao_vai_para_estudos() -> None:
    """A palavra adicionada deve acionar a regra de estudos."""
    assert category_by_rules("Preparar redação") == "estudos"
```

Esse bloco é um modelo de personalização, não um teste que já passa na versão
original. Use `python -m pytest tests/test_minha_rotina.py`. Primeiro observe a
falha, depois edite a regra e execute de novo. Uma expectativa deve refletir o
comportamento desejado, não repetir a mesma implementação por dentro.

Cadastre três tarefas fictícias, corrija uma prioridade, filtre uma categoria,
selecione uma tarefa pelo nome para concluir e reinicie. Registre o estado esperado e o observado. Teste também
uma categoria inválida e confirme que o arquivo não muda.

## Nível 3 — investigação (se houver tempo)

Compare uma frase com palavra conhecida, uma com sinônimo fora das regras e uma
ambígua. Registre categoria das regras, do ML e, quando disponível, do LLM.
Indique qual sugestão você aceitaria e por quê. Não declare vencedor com três
frases escolhidas depois de ver as saídas.

Proponha novos exemplos de desenvolvimento e um teste final separado. Como evitar
repetir o mesmo texto em treino e teste? O que aconteceria se houvesse muito mais
exemplos de casa que de voluntariado? Que pessoas ou modos de escrever poderiam
ficar pouco representados? Não acrescente dados pessoais para tentar resolver isso.

[Soluções e orientações do instrutor](solucoes-instrutor/README.md)
