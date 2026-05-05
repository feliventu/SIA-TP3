import numpy as np

from src.layers.base import Layer


class Activation(Layer):


    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:

        self._input = input_data
        return self._activate(input_data)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Aplica la chain rule: dL/dZ = dL/dA * dA/dZ = output_gradient * σ'(Z)
        """
        return output_gradient * self._activate_prime(self._input)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """La función de activación σ(z). Implementar en subclase."""
        raise NotImplementedError

    def _activate_prime(self, z: np.ndarray) -> np.ndarray:
        """La derivada de la activación σ'(z). Implementar en subclase."""
        raise NotImplementedError