"""Regressão linear univariada implementada sem framework de Machine Learning.

O módulo mantém previsão, derivação e atualização de parâmetros visíveis. A simplicidade de um
único atributo permite relacionar cada linha de Python às equações do modelo e do MSE.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .losses import mean_squared_error


@dataclass
class LinearRegression1D:
    """Modelo ``y_hat = peso * x + viés`` treinado por gradiente descendente em lote.

    Attributes:
        weight: Inclinação da reta. Indica quanto a previsão muda quando a entrada aumenta uma
            unidade, mantendo a convenção de unidades usada pelos dados.
        bias: Intercepto da reta. É a previsão matemática em ``x = 0``, embora esse ponto possa não
            possuir interpretação prática no domínio observado.

    Note:
        A classe aceita parâmetros iniciais para permitir carregamento de artefatos e experimentos.
        Valores padrão iguais a zero representam o estado antes do treinamento.
    """

    weight: float = 0.0
    bias: float = 0.0

    def predict_one(self, feature: float) -> float:
        """Calcula a saída da reta para uma entrada.

        Args:
            feature: Valor escalar do único atributo ``x``.

        Returns:
            Previsão ``weight * feature + bias``. O método não altera parâmetros e não treina o
            modelo; ele apenas realiza inferência com o estado atual.
        """
        return self.weight * float(feature) + self.bias

    def predict(self, features: Sequence[float]) -> list[float]:
        """Aplica :meth:`predict_one` a uma sequência não vazia de entradas.

        A ordem das previsões corresponde exatamente à ordem das entradas.

        Raises:
            ValueError: Se nenhuma entrada for fornecida.
        """
        if not features:
            raise ValueError("Forneça pelo menos uma entrada para previsão.")
        return [self.predict_one(feature) for feature in features]

    def gradients(
        self, features: Sequence[float], targets: Sequence[float]
    ) -> tuple[float, float]:
        """Calcula as derivadas do MSE em relação ao peso e ao viés.

        Para ``y_hat_i = w*x_i + b`` e ``MSE = (1/n)Σ(y_hat_i-y_i)²``, as derivadas são:

        - ``dMSE/dw = (2/n)Σ[(y_hat_i-y_i)*x_i]``;
        - ``dMSE/db = (2/n)Σ[y_hat_i-y_i]``.

        Args:
            features: Entradas ``x_i``.
            targets: Alvos ``y_i`` correspondentes, na mesma ordem.

        Returns:
            Par ``(gradiente_do_peso, gradiente_do_vies)`` calculado sobre todo o lote.

        Raises:
            ValueError: Se as entradas estiverem vazias ou os tamanhos forem diferentes.
        """
        if not features or len(features) != len(targets):
            raise ValueError("Entradas e alvos devem ter o mesmo tamanho não nulo.")
        # As previsões precisam refletir os mesmos parâmetros para todos os exemplos deste lote.
        predictions = self.predict(features)
        count = len(features)
        weight_gradient = (2 / count) * sum(
            (prediction - target) * feature
            for feature, target, prediction in zip(features, targets, predictions, strict=True)
        )
        bias_gradient = (2 / count) * sum(
            prediction - target
            for target, prediction in zip(targets, predictions, strict=True)
        )
        return weight_gradient, bias_gradient

    def fit(
        self,
        features: Sequence[float],
        targets: Sequence[float],
        *,
        learning_rate: float = 0.01,
        epochs: int = 1000,
    ) -> list[float]:
        """Ajusta os dois parâmetros por gradiente descendente em lote.

        Uma época calcula gradientes usando todos os exemplos, atualiza ``weight`` e ``bias`` e
        registra o MSE após a atualização. A taxa controla o tamanho do passo, não a direção.

        Args:
            features: Sequência não vazia de entradas de treino.
            targets: Alvos correspondentes às entradas.
            learning_rate: Multiplicador positivo aplicado a cada gradiente.
            epochs: Quantidade positiva de atualizações completas.

        Returns:
            Histórico com um MSE por época. A queda é evidência de otimização nos dados de treino,
            não prova de generalização em dados novos.

        Raises:
            ValueError: Se taxa ou épocas não forem positivas, ou se entradas e alvos violarem o
                contrato validado por :meth:`gradients`.

        Note:
            O método modifica a própria instância. Em atributos com escalas muito diferentes, a
            padronização anterior pode tornar a otimização numericamente mais estável.
        """
        if learning_rate <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")
        if epochs <= 0:
            raise ValueError("O número de épocas deve ser positivo.")
        history: list[float] = []
        for _ in range(epochs):
            weight_gradient, bias_gradient = self.gradients(features, targets)

            # Caminhar contra o gradiente busca reduzir localmente a função de perda.
            self.weight -= learning_rate * weight_gradient
            self.bias -= learning_rate * bias_gradient

            # Medimos depois do passo para que cada item descreva o estado recém-atualizado.
            history.append(mean_squared_error(targets, self.predict(features)))
        return history
