"""
Red neuronal como secuencia de capas.
Forward recorre las capas en orden, backward en reversa.
"""

import numpy as np
import json
from typing import List, Optional

from src.layers.base import Layer
from src.layers.linear import LinearLayer
from src.activations import ReLU, Tanh, Softmax
from src.losses.base import Loss


# Registry de activaciones para construir desde config
ACTIVATIONS = {
    "relu": ReLU,
    "tanh": Tanh,
    "softmax": Softmax,
}


class NeuralNetwork:
    """MLP secuencial: lista de capas que se ejecutan en orden."""

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    @classmethod
    def from_config(cls, architecture: List[int], activation: str = "relu",
                    output_activation: str = "same",
                    backend: str = "cpu") -> "NeuralNetwork":
        """
        Construye la red desde una configuración.

        Args:
            architecture: ej [784, 128, 64, 10] -> 3 capas lineales
            activation: activación para capas ocultas ("relu" o "tanh")
            output_activation: "same" = igual que ocultas, "softmax"/"relu"/"tanh", "none" = sin activación
            backend: "cpu" (numpy), "cuda" (CUDA cores), "tensor" (Tensor Cores)
        """
        act_cls = ACTIVATIONS[activation]
        layers = []

        for i in range(len(architecture) - 1):
            layers.append(LinearLayer(architecture[i], architecture[i + 1], backend=backend))

            if i < len(architecture) - 2:
                layers.append(act_cls())
            else:
                # Última capa
                if output_activation == "same":
                    layers.append(act_cls())
                elif output_activation != "none":
                    layers.append(ACTIVATIONS[output_activation]())

        return cls(layers)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante: X -> capa1 -> capa2 -> ... -> output"""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad: np.ndarray) -> None:
        """Propagación hacia atrás: gradiente fluye de la loss a la primera capa."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self, learning_rate: float) -> None:
        """Actualiza pesos con SGD vanilla."""
        for layer in self.layers:
            layer.update_params(learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass sin guardar estado (para inferencia)."""
        return self.forward(X)

    def save(self, path: str, epoch: int = 0) -> None:
        """Guarda los pesos de la red en un archivo .npz"""
        params = {"epoch": np.array(epoch)}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                params[f"W_{i}"] = layer.W
                params[f"b_{i}"] = layer.b
        np.savez(path, **params)

    def load(self, path: str) -> int:
        """Carga pesos desde un archivo .npz y retorna la época guardada."""
        data = np.load(path)
        epoch = int(data["epoch"]) if "epoch" in data else 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                layer.W = data[f"W_{i}"]
                layer.b = data[f"b_{i}"]
        return epoch
