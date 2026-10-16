# Referências oficiais

[← Guia](README.md)

Consultadas em 16 de outubro de 2026. São fontes técnicas, não promessa de
qualidade das sugestões ou de desempenho nos computadores da turma.

- [Python: csv](https://docs.python.org/3/library/csv.html): leitura, escrita,
  cabeçalho e quoting de campos; usado na persistência local e nos exemplos.
- [Gradio: Blocks](https://www.gradio.app/docs/gradio/blocks): componentes, eventos
  e servidor local. A prática declara `gradio>=5.49,<6`; versões futuras da
  documentação podem divergir dessa API. A validação registra a versão executada.
- [Ollama: API de chat](https://docs.ollama.com/api/chat): mensagens, formato JSON,
  `stream`, `think` e resposta. O projeto usa HTTP da biblioteca padrão Python.
- [Ollama: saídas estruturadas](https://ollama.com/blog/structured-outputs): uso
  de esquema JSON; a aplicação mantém sua validação determinística adicional.
- [Qwen3 1.7B](https://ollama.com/library/qwen3:1.7b) e
  [Qwen3 0.6B](https://ollama.com/library/qwen3:0.6b): tags locais e informações
  de download; tamanho de arquivo não é orçamento completo de memória.
- [Scikit-learn: extração de atributos textuais](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
  representação por contagens e limites de saco de palavras.
- [Scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes):
  hipótese de independência condicional e suavização no modelo multinomial.

A base de exemplos deste projeto foi escrita para ensino, sem coleta de dados
pessoais ou garantia de representatividade. Qwen3 tem licença própria dos pesos;
o código didático mantém a licença do repositório. Não redistribuímos pesos.
