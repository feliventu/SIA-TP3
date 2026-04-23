"""
Tanh (Tangente Hiperbólica).

== Fórmula ==
  forward:  σ(z) = tanh(z) = (e^z - e^-z) / (e^z + e^-z)
  backward: σ'(z) = 1 - tanh²(z)

"""

import numpy as np

from .base import Activation


class Tanh(Activation):
    """Tanh: salida en [-1, 1], centrada en 0."""

    def _activate(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def _activate_prime(self, z: np.ndarray) -> np.ndarray:
        t = np.tanh(z)
        return 1.0 - t ** 2
