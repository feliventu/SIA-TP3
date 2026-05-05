"""
SGD con soporte opcional de Momentum.

Sin momentum: W -= lr * dW
Con momentum:
  v = beta * v + dW
  W -= lr * v
"""

from typing import List, Dict
import numpy as np

from src.layers.base import Layer
from .base import Optimizer


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.lr = learning_rate
        self.momentum = momentum
        # Velocidades por parámetro (se inicializan en el primer step)
        self._velocities: Dict[int, List[np.ndarray]] = {}

    def step(self, layers: List[Layer]) -> None:
        for i, layer in enumerate(layers):
            params = layer.parameters()
            grads = layer.gradients()

            if len(params) == 0:
                continue

            # Inicializar velocidades la primera vez
            if i not in self._velocities:
                self._velocities[i] = [np.zeros_like(p) for p in params]

            for j, (param, grad) in enumerate(zip(params, grads)):
                if self.momentum > 0:
                    self._velocities[i][j] = self.momentum * self._velocities[i][j] + grad
                    param -= self.lr * self._velocities[i][j]
                else:
                    param -= self.lr * grad
