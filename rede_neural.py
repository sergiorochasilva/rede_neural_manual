import random
from neuronio import Neuronio


class RedeNeuralDensa:
    camadas: list[list[Neuronio]]

    def __init__(
        self,
        tam_entrada: int,
        tam_saida: int,
        hidden_layers: int,
        hidden_size: int,
        activation_function: str = "relu",
    ):
        super().__init__()
        saida = [Neuronio(hidden_size, "linear") for _ in range(tam_saida)]
        h0 = [Neuronio(tam_entrada, activation_function) for _ in range(hidden_size)]
        if hidden_layers > 1:
            hidden = [
                [Neuronio(hidden_size, activation_function) for _ in range(hidden_size)]
                for _ in range(hidden_layers - 1)
            ]
            self.camadas = [h0] + hidden + [saida]
        else:
            self.camadas = [h0] + [saida]

    def forward(self, entrada: list[float]) -> list[float]:
        entradas_camada = entrada  # vetores puros
        for camada in self.camadas:
            saidas = []
            for neuronio in camada:
                neuronio.entrada = entradas_camada
                saidas.append(neuronio.forward())
            entradas_camada = saidas

        return entradas_camada

    def treinar(
        self,
        entradas: list[list[float]],
        saidas: list[list[float]],
        epochs: int = 1000,
        learning_rate: float = 0.01,
    ):
        """
        Método para treinar a rede neural com as entradas e saídas fornecidas.
        Este método deve implementar o algoritmo de retropropagação e ajuste dos pesos.
        """

        for epoch in range(epochs):
            if epoch % 100 == 0:
                # print(f"Saída de amostra: {self.forward(entradas[0])}")
                total_loss = 0
                for entrada, saida in zip(entradas, saidas):
                    previsao = self.forward(entrada)
                    total_loss += sum(
                        (s - p) ** 2 for s, p in zip(saida, previsao)
                    ) / len(saida)
                print(f"Epoch {epoch}/{epochs}, MSE: {total_loss / len(entradas)}")

            dados = list(zip(entradas, saidas))
            random.shuffle(dados)
            entradas, saidas = zip(*dados)

            for entrada, saida in zip(entradas, saidas):
                previsao = self.forward(entrada)

                erro = [p - s for s, p in zip(saida, previsao)]

                # gradiente = [2 / len(saida) * erro[i] for i in range(len(saida))]
                gradiente = [
                    (2 / len(saida))
                    * erro[i]
                    * self.camadas[-1][i].activation_derivada(self.camadas[-1][i].z)
                    for i in range(len(saida))
                ]

                gradientes_por_camada = [gradiente]

                for i in range(len(self.camadas) - 2, 0, -1):
                    camada = self.camadas[i]
                    proxima_camada = self.camadas[i + 1]
                    gradientes_nova_camada = []

                    for j, neuronio in enumerate(camada):
                        grad = 0.0

                        for k, neuronio_prox in enumerate(proxima_camada):
                            grad += gradientes_por_camada[0][k] * neuronio_prox.pesos[j]

                        grad = grad * neuronio.activation_derivada(neuronio.z)
                        gradientes_nova_camada.append(grad)
                    gradientes_por_camada.insert(0, gradientes_nova_camada)

                for i in range(1, len(self.camadas)):
                    camada = self.camadas[i]
                    entradas_camada = [n.saida for n in self.camadas[i - 1]]

                    for j, neuronio in enumerate(camada):
                        for k in range(len(neuronio.pesos)):
                            neuronio.pesos[k] -= (
                                learning_rate
                                * gradientes_por_camada[i - 1][j]
                                * entradas_camada[k]
                            )
                            # neuronio.pesos[k] = max(min(neuronio.pesos[k], 3), -3)
                        neuronio.bias -= learning_rate * gradientes_por_camada[i - 1][j]
                        # neuronio.bias = max(min(neuronio.bias, 3), -3)


if __name__ == "__main__":
    # Exemplo de uso da Rede Neural Densa
    rede = RedeNeuralDensa(3, 2, 2, 4, activation_function="tanh")

    entradas = [
        [0.5, 0.2, 0.1],
        [0.1, 0.3, 0.5],
        [0.7, 0.2, 0.3],
        [0.4, 0.6, 0.1],
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.3],
        [0.3, 0.3, 0.4],
        [0.6, 0.2, 0.5],
        [0.1, 0.8, 0.2],
        [0.4, 0.4, 0.2],
        # 490 entradas adicionais coerentes:
    ] + [[(i % 10) / 10, (i % 20) / 20, (i % 25) / 25] for i in range(10, 500)]

    saidas = [
        [0.1, 0.9],
        [0.3, 0.7],
        [0.5, 0.5],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.4, 0.6],
        # 490 saídas adicionais coerentes:
    ] + [
        [
            sum([(i % 10) / 10, (i % 20) / 20, (i % 25) / 25]) / 3,  # média normalizada
            1.0 if sum([(i % 10) / 10, (i % 20) / 20, (i % 25) / 25]) > 1.0 else 0.0,
        ]
        for i in range(10, 500)
    ]

    entradas = [[x * 2 - 1 for x in entrada] for entrada in entradas]
    saidas = [[s * 2 - 1 for s in y] for y in saidas]
    rede.treinar(entradas, saidas, epochs=10000, learning_rate=0.001)

    entrada_teste = [0.5, 0.2, 0.1]
    entrada_teste = [x * 2 - 1 for x in entrada_teste]  # Normalizando a entrada

    print(saidas[0])
    print(rede.forward(entrada_teste))
