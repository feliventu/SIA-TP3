"""
ReLU (Rectified Linear Unit).

  forward:  σ(z) = max(0, z)
  backward: σ'(z) = 1 si z > 0, 0 si z ≤ 0

"""

import numpy as np

from .base import Activation


class ReLU(Activation):
    """ReLU: max(0, z)"""

    def _activate(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def _activate_prime(self, z: np.ndarray) -> np.ndarray:
        # Devuelve 1.0 donde z > 0, 0.0 donde z ≤ 0
        # astype(float) convierte el bool array a 0.0/1.0
        return (z > 0).astype(float)