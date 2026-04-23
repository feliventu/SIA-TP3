"""
Adam (Adaptive Moment Estimation).

m = β1*m + (1-β1)*g        primer momento (media del gradiente)
v = β2*v + (1-β2)*g²       segundo momento (varianza del gradiente)
m̂ = m / (1-β1^t)           corrección de bias
v̂ = v / (1-β2^t)
W -= lr * m̂ / (√v̂ + ε)
"""

from typing import List, Dict
import numpy as np

from src.layers.base import Layer
from .base import Optimizer


class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep (para bias correction)
        self._m: Dict[int, List[np.ndarray]] = {}  # primer momento
        self._v: Dict[int, List[np.ndarray]] = {}  # segundo momento

    def step(self, layers: List[Layer]) -> None:
        self.t += 1

        for i, layer in enumerate(layers):
            params = layer.parameters()
            grads = layer.gradients()

            if len(params) == 0:
                continue

            # Inicializar momentos la primera vez
            if i not in self._m:
                self._m[i] = [np.zeros_like(p) for p in params]
                self._v[i] = [np.zeros_like(p) for p in params]

            for j, (param, grad) in enumerate(zip(params, grads)):
                # Actualizar momentos
                self._m[i][j] = self.beta1 * self._m[i][j] + (1 - self.beta1) * grad
                self._v[i][j] = self.beta2 * self._v[i][j] + (1 - self.beta2) * (grad ** 2)

                # Corrección de bias
                m_hat = self._m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self._v[i][j] / (1 - self.beta2 ** self.t)

                # Actualizar parámetro
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
