"""
Softmax

== Fórmula ==
  forward:  σ(z_i) = e^z_i / Σ_j e^z_j

"""

import numpy as np

from .base import Activation


class Softmax(Activation):
    """Softmax: convierte logits en probabilidades. Solo para capa de salida."""

    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Softmax.
        Input: (n_classes, m_samples)
        Output: (n_classes, m_samples), cada columna suma 1
        """
        self._input = input_data

        z_shifted = input_data - np.max(input_data, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        self._output = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        return self._output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Pass-through: el gradiente combinado Softmax+CrossEntropy
        """
        return output_gradient
