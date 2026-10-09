# Plano baseado na inspeção de 2 de outubro de 2026

## Estado encontrado antes da expansão

A árvore Git estava limpa. Havia sete diretórios conceituais completos até
classificação, 23 fontes e 76 testes anteriores. Não existiam métricas completas,
neurônio, camada densa, MLP, backpropagation, modelos PyTorch ou projeto integrador.
AGENTS.md correspondia à implementação; seu registro de ambiente não descrevia
esta cópia: `.venv`, dados processados e PNGs estavam ausentes e precisaram ser
recriados. `docs/referencias.md` não existe; o documento canônico é
`docs/referencias-gerais.md`. O extra `torch>=2.2,<3` já estava declarado.

A inspeção cobriu instruções, README, jornada, metodologia, referências/glossário,
configuração, árvore conceitual, exercícios progressivos, implementação de sensores,
logística Python/NumPy/sklearn, CLIs, persistência, operações manuais e testes.
A validação integrada confirmou os resultados anteriores de regressão e classificação.

## Reuso e fases

1. Métricas em Python puro reutilizam validação de rótulos e limiar inclusivo.
2. Partição tripla reutiliza `train_test_indices`; gerador e scalers permanecem.
3. Neurônio reutiliza produto escalar e sigmoid; ReLU amplia ativações sem migrar API.
4. Camada e MLP NumPy têm formas explícitas; a saída conserva logits.
5. Backpropagation reutiliza BCE estável; derivar, verificar e atualizar são funções distintas.
6. Deep Learning conecta composição a CNN/RNN/Transformer sem modelos grandes.
7. PyTorch replica parâmetros e gradientes, mantendo dependência opcional por etapa.
8. Persistência/inferência preservam arquitetura, parâmetros, ordem das colunas e escalas.
9. Decisão estruturada local organiza probabilidade, recomendação, política e revisão humana.
10. Jev usa fontes oficiais consultadas na data; SDK remoto é exercício opcional separado.
11. Engenharia e projeto integrador unem fluxo completo, testes, observabilidade e rubrica.

Diretórios novos só são justificados por material e práticas reais. README, jornada,
glossário e AGENTS recebem registros por fase; o registro consolidado substitui
estados planejados somente após execução. Soluções ficam em `solucoes-instrutor`.

## Dependências, recursos e verificação

Base continua Python >=3.11; estudantes usam 3.11/3.12. O extra torch permanece opcional por etapa, com mínimo elevado de 2.2 para 2.10
por aviso oficial GHSA-63cw-57p8-fm3p relacionado à recarga weights_only.
PyTorch é instalado no
momento da trilha que o exige, com instruções CPU específicas por plataforma.
O ambiente executado usa Python 3.14.7; isso não prova compatibilidade de todas
as versões mínimas. Jev/ExecuTorch não entram na validação obrigatória.
Não se introduz download obrigatório de modelo pré-treinado nem dependência local-ai.

400 linhas e rede 2→4→1 mantêm matrizes pequenas. Testes verificam contas,
partições sem vazamento, formas, todos os gradientes, equivalência NumPy/PyTorch,
queda de perda, restauração, persistência, processo novo, política e abstenção.
Ruff/mypy e execução da prática antecedem o próximo passo. Gráficos são renderizados
em CPU e inspecionados. O custo do runtime pode exceder amplamente o dos parâmetros:
medir latência, bytes e tamanho do artefato em vez de prometer rapidez universal.

O modelo final não controla equipamento. Produz saída didática com política explícita,
limites e revisão; nenhum dado sintético valida manutenção ou segurança industrial.
