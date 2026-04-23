from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Transforma la entrada en salida (propagación hacia adelante).

        Args:
            input_data: array de shape (n_input, m_samples)

        Returns:
            array de shape (n_output, m_samples)
        """
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Calcula el gradiente respecto a la entrada, dado el gradiente
        respecto a la salida (propagación hacia atrás).

        Args:
            output_gradient: dL/dOutput, shape (n_output, m_samples)

        Returns:
            dL/dInput, shape (n_input, m_samples)
        """
        pass

    def parameters(self) -> List[np.ndarray]:
        """
        Devuelve la lista de parámetros entrenables (ej: [W, b]).
        Las capas sin parámetros (activaciones) devuelven lista vacía.
        """
        return []

    def gradients(self) -> List[np.ndarray]:
        """
        Devuelve los gradientes de los parámetros, en el mismo orden
        que parameters(). Se llenan después de llamar a backward().
        """
        return []

    def update_params(self, learning_rate: float) -> None:
        pass
