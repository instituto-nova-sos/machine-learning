# Visão geral

[← Voltar à aula principal](../README.md) · [Jornada de aprendizado →](jornada-de-aprendizado.md)

O curso trata aprendizado de máquina como encontro entre modelagem matemática, computação,
evidência empírica e engenharia de software. Uma previsão é a saída de uma função parametrizada,
não uma verdade. Dados carregam decisões de coleta; métricas carregam escolhas de prioridade.

```mermaid
flowchart LR
  IA[Inteligência Artificial] -->|inclui| SIM[Abordagens sem aprendizado de máquina]
  IA -->|inclui| ML[Aprendizado de Máquina]
  ML -->|inclui| OUTROS[Regressão, árvores e outros métodos]
  ML -->|inclui| DL[Aprendizado Profundo]
  DL -->|inclui| LLM[LLMs modernos]
```

As setas significam “inclui”, não equivalência. Assim, `DL ⊂ ML ⊂ IA`. Existem sistemas de IA
que não usam ML e métodos de ML que não usam redes profundas. Um LLM moderno não é um campo no
mesmo nível: é uma família de modelos de linguagem implementada com redes neurais profundas,
normalmente Transformers.
