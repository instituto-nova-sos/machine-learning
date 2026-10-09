# PyTorch: automatizar operações que já compreendemos

[← Deep Learning](../deep-learning/README.md)

Pré-requisito: reproduzir os gradientes NumPy da MLP. Objetivo: reconhecer o papel
de cada linha, treinar em CPU e usar pesos salvos em um novo processo.

## Instalação por etapa

O extra `torch` já existia; seu mínimo foi elevado de 2.2 para 2.10 porque
passamos a ensinar recarga de state_dict. O [aviso oficial GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
registra falha em weights_only nas versões até 2.9.1, corrigida em 2.10.
Isso não torna arquivos desconhecidos confiáveis. A base `.[dev]` continua suficiente até backpropagation
e para o caminho NumPy do projeto final. Para completar **esta etapa**, instale
PyTorch localmente. É dependência de software, não serviço remoto ou conta paga.
Recomendamos Python 3.11/3.12; wheels variam por Python, sistema e arquitetura.
Consulte a matriz de compatibilidade para o Python e a plataforma escolhidos;
a versão mínima declarada não garante wheel para qualquer combinação.

Linux/Windows com pip e CPU:

```bash
python -m pip install "torch>=2.10,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

No macOS, use PyPI e mantenha nossos exemplos em CPU:

```bash
python -m pip install -e ".[dev,torch]"
```

Confira o [seletor oficial](https://pytorch.org/get-started/locally/), principalmente
em máquinas antigas. O wheel usado aqui (macOS ARM64, Python 3.14, torch 2.14.1)
teve download de aproximadamente 127 MB, além de dependências. O tamanho varia;
instalação/runtime ocupam muito mais que os 17 parâmetros da rede. Não instalamos
TorchVision, TorchAudio, CUDA, ExecuTorch ou pesos de LLM. A validação registrada
não cobre outros sistemas nem Python mínimo; não é garantia de wheel universal.

## Lista → ndarray → tensor

Uma lista contém objetos Python; ndarray representa um arranjo numérico com
forma e dtype; Tensor acrescenta operações e integração com dispositivos/autograd.
Tensor pode ser escalar (zero eixos), vetor, matriz ou ter mais eixos.

```python
import numpy as np
import torch

values = [[1., 2.], [3., 4.]]
array = np.asarray(values, dtype=np.float64)
tensor = torch.tensor(values, dtype=torch.float64, device="cpu")
print(array.shape, tensor.shape, tensor.dtype, tensor.device)
print(tensor[0], tensor[:, 1], tensor @ tensor.T)
```

Forma `(2,2)` não explica significado: mantenha contrato de colunas. float64 facilita
comparar diferenças finitas; float32 costuma economizar memória, mas exige outras
tolerâncias. `.to(device)` muda onde operações/armazenamento ocorrem, não a tarefa.
Não precisamos de GPU para quatro unidades ocultas.

## Autograd no mesmo escalar

```python
import torch

w = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
z = 2*w+b
a = z**2
loss = (a-1)**2/2
loss.backward()
print(z.item(), a.item(), loss.item(), w.grad.item(), b.grad.item())
```

Compare com `scalar_example`: z=1,1, a=1,21, perda=0,02205,
gradientes 0,924 e 0,462. `requires_grad` pede rastreamento; forward registra
operações; `backward` aplica regras locais em sentido inverso e acumula gradientes.
Não aprende uma fórmula de derivada por adivinhação e não atualiza w por si só.

## Module, parâmetros e Linear

`nn.Parameter` é um tensor registrado como parâmetro treinável de um Module.
No modelo mínimo poderíamos escrever:

```python
import torch
from torch import nn

class LinearManual(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 1, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, X):
        return X @ self.weight + self.bias

model = LinearManual()
print(model(torch.tensor([[1., 2.]], dtype=torch.float64)))
```

`nn.Linear(2,4)` automatiza essa soma para quatro saídas. Guarda W como `(4,2)`,
de modo que seu cálculo equivale a `X @ weight.T + bias`. A transposta é essencial
para copiar nossa W NumPy `(2,4)`. ReLU corresponde à mesma máscara que derivamos.
[EquipmentMLP](../src/sos_ml/torch_models/equipment.py) usa essas duas camadas.
Os testes comparam todos os pesos/vieses e gradientes, não apenas acertos finais.

## Perdas e otimização

`nn.MSELoss()` calcula média `(predição−alvo)²`, como nosso MSE. BCE é outra perda:
`nn.BCEWithLogitsLoss()` recebe logits e alvos float da **mesma forma (n,1)**.
Combinar sigmoid e logaritmos permite uma expressão numericamente estável como
na logística manual. Não aplique sigmoid antes dessa API: seria sigmoid duas vezes.
Sigmoid continua necessária para apresentar probabilidade ao usuário.

```python
import torch
from torch import nn
from sos_ml.torch_models.equipment import EquipmentMLP

model = EquipmentMLP()
X = torch.tensor([[-1., -1.], [1., 1.]], dtype=torch.float64)
y = torch.tensor([[0.], [1.]], dtype=torch.float64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(3):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(X)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())
```

X/y são o lote e alvos; model(X) é forward; loss compara saída e alvo;
zero_grad limpa derivadas anteriores, pois backward acumula; backward calcula;
step aplica `θ -= 0,1*gradiente` no SGD sem momentum aqui. Taxa/épocas são escolhas,
não parâmetros aprendidos. A perda impressa nesse bloco é **antes** do passo;
`train_step` retorna perda depois dele: compare momentos equivalentes.
Calcular métricas envolve sigmoid/limiar e conjunto adequado, não o gradiente.

## Treino, avaliação e ausência de grafo

`train()` e `eval()` mudam comportamento de camadas: dropout desativa aleatoriamente parte das
ativações no treino por uma máscara (zeros descartam, valores positivos preservam
e reescalam a média); na avaliação preserva todas as ativações; BatchNorm normalmente usa estatísticas de
lote no treino e acumuladas na avaliação. Nossa rede não usa essas duas camadas,
mas preserva os modos. `eval()` **não desliga gradientes**. `no_grad()` desliga
rastreamento no bloco. São decisões independentes. Também existe `inference_mode`,
com restrições adicionais; usamos no_grad pela simplicidade desta ponte didática.

## Treino completo e persistência

```bash
python -m sos_ml.torch_demo treinar
python -m sos_ml.torch_demo inferir
python -m pytest tests/test_torch_foundations.py tests/test_equipment_persistence.py
```

O primeiro comando usa 240/80/80 exemplos, escala só no treino, seleciona época pela
BCE de validação e restaura os melhores pesos. Depois mede teste com limiar fixo 0,5.
O segundo é **outro processo**: reconstrói arquitetura, carrega parâmetros/escalas,
chama eval, desliga gradientes e calcula só inferência para [80 °C,5 mm/s].
Não gera dataset nem chama fit. Compare com a logística, cujo domínio é o mesmo,
mas partição e família diferem; não atribua diferença só à biblioteca.

Arquitetura é o código de operações; `state_dict` mapeia nomes a parâmetros e
buffers. O arquivo `.pt` guarda esse dicionário, não a instância da classe.
O JSON correspondente guarda arquitetura/ordem das colunas/scalers e uma cópia
inspecionável dos parâmetros para conferir equivalência e permitir runtime NumPy.
Os dois arquivos são comparados ao recarregar: não misture versões.

Um checkpoint de treino também precisa estado do otimizador, época, sementes e
metadados; nosso artefato é de **inferência**, não retoma exatamente um treino.
Usamos `torch.load(..., map_location="cpu", weights_only=True)`, validamos tamanho,
chaves, formas/dtype e finitude. Nunca use `weights_only=False` para contornar erro
de arquivo desconhecido. Carregar pickle de origem não confiável pode executar
código; weights_only reduz riscos, não autentica origem nem garante recursos limitados.

Sementes e equivalência numérica validam esta implementação, não probabilidades
calibradas ou uso industrial. A rede sintética não controla máquinas reais.

[Exercícios](exercicios.md) · [Referências](referencias.md) ·
[Próximo: inferência local](../inferencia-local/README.md)
