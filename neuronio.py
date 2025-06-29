import math
import random


class Neuronio:
    entrada: list[float]
    pesos: list[float]
    bias: float
    z: float
    saida: float

    def __init__(self, conexoes: int, activation_function: str = "relu"):
        self.entrada = [0.0] * conexoes  # Inicializa as entradas com zeros

        limite = 1 / math.sqrt(conexoes) if conexoes > 0 else 1
        self.pesos = [random.uniform(-limite, limite) for _ in range(conexoes)]

        self.bias = random.uniform(-0.1, 0.1)
        self.activation_function = activation_function

    def forward(self) -> float:
        """
        Calcula a saída do neurônio com base nas entradas, pesos e bias.
        A saída é calculada como a soma ponderada das entradas mais o bias.
        """
        soma = sum(e * p for e, p in zip(self.entrada, self.pesos))
        self.z = soma + self.bias
        self.saida = self.activation(self.z)
        return self.saida

    def activation(self, x: float) -> float:
        """
        Função de ativação do neurônio.
        Nese caso será uma simples sigmoide.
        A função sigmoide é definida como 1 / (1 + exp(-x)).
        """
        if self.activation_function == "sigmoid":
            return 1.0 / (1.0 + math.exp(-x))
        elif self.activation_function == "tanh":
            return math.tanh(x)
        elif self.activation_function == "linear":
            return x
        else:
            # ReLU (Rectified Linear Unit)
            return max(0.0, x)

    def activation_derivada(self, x: float) -> float:
        if self.activation_function == "sigmoid":
            sig = 1.0 / (1.0 + math.exp(-x))
            return sig * (1 - sig)
        elif self.activation_function == "tanh":
            t = math.tanh(x)
            return 1 - t * t
        elif self.activation_function == "linear":
            return 1
        else:
            return 1.0 if x > 0 else 0.0  # ReLU
