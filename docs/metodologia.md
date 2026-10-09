# Metodologia

[← Voltar à aula principal](../README.md) · [Glossário →](glossario.md)

Cada ciclo combina intuição explicitamente marcada, notação, cálculo pequeno, implementação,
teste e reflexão crítica. Demonstrações começam com Python puro, avançam para vetorização e só
então usam bibliotecas de alto nível. Analogias são pontes temporárias: por exemplo, imaginar a
inclinação como uma “rampa” ajuda a visualizar a derivada, mas uma derivada é formalmente o
limite de uma razão de variações.

Avaliação considera explicação, cálculo, código, interpretação, reprodutibilidade e ética — não
somente uma métrica preditiva.


Na continuação, a avaliação é feita por etapas: cálculo manual → contrato de forma
→ comparação numérica → gradientes → treino/validação → inferência em processo novo
→ política. Um resultado tipado ainda pode estar errado; uma taxa alta ainda pode
esconder falhas ou baixa cobertura. Exercícios incluem recusa de automação.

O domínio sintético familiar reduz a troca de contexto, mas não demonstra qualidade
industrial. Compare as famílias em protocolos iguais antes de atribuir diferenças
a arquitetura. Testes e bibliotecas automatizam conferências, não escolhem finalidade,
custos, grupos, limites ou responsabilidade por nós.

O estudo Jev tem duas camadas: arquitetura local obrigatória e integração oficial
remota opcional. Acesso pago não participa de rubrica. A etapa PyTorch é executada
em CPU com dependência instalada por etapa, depois da derivação manual.
