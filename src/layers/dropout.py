"""
Capa de Dropout para regularización.
Implementa 'Inverted Dropout', que escala las activaciones durante el entrenamiento
para mantener el valor esperado constante sin modificar la inferencia.
"""

import numpy as np

from src.layers.base import Layer


class DropoutLayer(Layer):
    """
    Inverted Dropout Layer.
    """

    def __init__(self, drop_probability: float):
        """
        Args:
            drop_probability: probabilidad de "apagar" una neurona (ej: 0.2 para 20%)
        """
        self.p = drop_probability
        self.keep_prob = 1.0 - drop_probability
        self.mask = None

    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Aplica dropout solo si is_training=True.
        """
        if is_training and self.keep_prob > 0.0:
            # Genera matriz aleatoria de la misma forma, con 1s con probabilidad keep_prob
            # Se divide por keep_prob para mantener la escala esperada
            self.mask = (np.random.rand(*input_data.shape) < self.keep_prob) / self.keep_prob
            return input_data * self.mask
        else:
            self.mask = None
            return input_data

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Aplica la misma máscara al gradiente.
        """
        if self.mask is not None:
            return output_gradient * self.mask
        else:
            return output_gradient
