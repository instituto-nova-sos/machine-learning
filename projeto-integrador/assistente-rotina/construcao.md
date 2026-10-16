# Aula 1 — Construir e entender a aplicação funcionando

[← Guia de execução](README.md) · [Próximo encontro →](personalizacao.md)

**Duração total: 120 minutos, incluindo pausa.** Ambiente preparado antes da aula
conforme o guia. Pré-requisitos: ler uma função, um `if`, uma lista e um dicionário.
Objetivo observável: cadastrar duas tarefas, corrigir uma sugestão, filtrar,
selecionar uma tarefa para concluir e explicar o papel de regras, ML, LLM e validação.

## Roteiro com relógio

| Minutos | Atividade | Resultado esperado |
|---|---|---|
| 0–10 | Apresentar problema cotidiano e rodar exemplo pronto | duas tarefas na revisão, ainda sem arquivo salvo |
| 10–30 | Ler contrato e regras; simular decisões no papel | explicar entrada, separação, categoria e prioridade |
| 30–45 | Construir uma regra pequena e testar no terminal | mudança compreendida, antes de mexer na interface |
| 45–60 | Salvar, filtrar, concluir e reiniciar | CSV persiste e ID permanece estável |
| 60–70 | Pausa | dez minutos incluídos |
| 70–90 | Contar palavras e comparar com Scikit-learn | distinguir regra escrita de padrão aprendido |
| 90–105 | Demonstrar Ollama e fallback com resposta inválida | identificar origem real e necessidade de revisão |
| 105–120 | Exercício em dupla e explicação de saída | entrega mínima e diagnóstico de dúvidas |

Soma: 10 + 20 + 15 + 15 + 10 + 20 + 15 + 15 = **120 minutos**.
Se houver dificuldade de instalação, mantenha o ritmo em duplas com máquina pronta
ou use o núcleo no terminal; downloads e configuração não são atividades extras
a acrescentar ao encontro.

## 0–30: partir de uma tarefa concreta

Pergunte: “O que você precisa fazer amanhã?” Use exemplos fictícios, sem dados
pessoais. Digite `Pagar boleto hoje; estudar Python`. Antes de clicar em analisar,
anote quantas tarefas espera e quais prioridades deseja. Compare com a revisão.
Não aceite uma sugestão apenas porque a interface parece correta.

Leia `domain.py`: `Task` reúne os três campos de uma sugestão. `id` e `status`
pertencem à persistência, só depois de salvar. `ValueError` significa que a função
recusou uma entrada fora do contrato. Não precisa memorizar dataclasses: observe
as entradas e o que cada validação impede.

Leia `extract_by_rules()`. Percorra `split`, `strip`, a lista resultante e o `if`
que detecta palavras. Faça no papel: quais partes surgem em `Lavar roupa; estudar`?
A normalização remove acentos só para comparação; a descrição original permanece.
A regra não foi treinada. Teste `não é urgente`: por que a prioridade dá alta?

## 30–45: construir uma mudança pequena em Python

Em `rules.py`, acrescente `revisão` à tupla de palavras de estudos. Antes de editar,
rode o comando abaixo e registre a categoria; depois edite e rode novamente.

```bash
python -c "from sos_ml.rotina.rules import category_by_rules; print(category_by_rules('Fazer revisão'))"
```

Esperado: `outros` antes, `estudos` depois. Explique por que escrever uma palavra na
tupla não é treinamento. Reinicie a interface para carregar código alterado. Faça
esta mudança em sua cópia; o material de referência mantém as palavras originais.

## 45–60: observar um arquivo como estado da aplicação

Na revisão, mude a prioridade de estudar para `baixa`. Confirme e salve. Filtre
por estudos na aba **Minhas tarefas** e selecione a tarefa pelo nome para concluir. Encerre e abra de novo o servidor. A tarefa
continua concluída porque foi persistida em CSV, e não porque o modelo “lembrou”.

Com o app encerrado, abra o CSV em editor de texto e localize cabeçalho, ID e status.
Não altere o arquivo nesta etapa. Separe dois conceitos: sugestão ainda em memória
e registro confirmado em disco. Clique duas vezes em salvar: a mesma tarefa pendente
não deve ser duplicada. Filtros mudam a visualização, nunca a identidade da tarefa.

## 70–90: contar antes de usar a biblioteca

Considere duas categorias e o vocabulário `[boleto, estudar]`. Represente `estudar`
como `[0,1]` e `boleto boleto` como `[2,0]`. Agora abra `exemplos.csv` e encontre
texto, categoria e partição. O rótulo é fornecido por quem preparou os exemplos;
o algoritmo não descobre sozinho o significado das categorias.

Leia `train_classifier()`: a Pipeline ajusta contagens e Naive Bayes no treino.
A suavização evita probabilidades zero; a hipótese de independência simplifica
relações entre palavras. Execute `python -m sos_ml.rotina.classifier`. Conte acertos
e compare com 2/14 do baseline constante. Mesmo muitos acertos em 14 frases
inventadas não comprovam desempenho no bairro, trabalho ou família.

Na lateral, abra **Como o assistente funciona** e escolha **Regras + ML**. Compare `buscar oportunidade de trabalho`
com `enviar currículo`. O primeiro tem palavras do treino, mas não das regras de
emprego. Sem palavras conhecidas, o ML devolve `outros`. Mostre que prioridade e
separação ainda vêm do Python neste modo.

## 90–105: LLM, validação e fallback

Na máquina preparada, abra **Como o assistente funciona** e selecione **LLM local (Ollama)** e descreva duas ações numa
frase. Leia `llm.py`: mensagem do sistema, texto do usuário, esquema JSON e parser.
Gerar JSON com a forma certa não comprova a fidelidade ao pedido. Procure omissões,
prioridades exageradas e descrições inventadas antes de salvar.

Se não houver Ollama, use esse mesmo modo e observe o aviso de fallback. Se o
Ollama estiver disponível, o instrutor demonstra uma falha de contrato com o teste
abaixo, sem parar servidores de terceiros:

```bash
python -m pytest tests/test_rotina.py -k 'parse_rejects or fallback' -v
```

Os testes simulam respostas e falhas; isso não prova qualidade real do Qwen3.
A regra de retorno funciona sem servidor. Uma frase longa que as regras não
conseguem separar pode exigir reescrita com `;`, como informa a interface.

## 105–120: entrega mínima em dupla

Cada dupla entrega uma demonstração de até dois minutos com duas tarefas fictícias,
uma correção humana, um filtro e uma conclusão persistida após reinício. Uma pessoa
opera; a outra explica: “qual parte foi escrita por nós, qual foi aprendida e qual
validou os dados?”. Troquem os papéis.

Saída esperada: aplicativo funcionando e três frases no caderno distinguindo
regra, classificação e geração. Se um erro ocorrer, mostre a mensagem e como
corrigi-lo; não esconda falhas para aparentar qualidade. Guarde uma ideia de
personalização para o próximo encontro.
