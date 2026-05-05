"""
Cross-Entropy Loss .

== Fórmula ==
  L = -(1/m) · Σ_i Σ_j  y_ij · log(ŷ_ij)

Donde:
  - y es one-hot encoded: cada columna tiene un 1 en la clase correcta
  - ŷ es la salida del softmax (probabilidades)
  - La suma recorre clases (i) y muestras (j)

"""

import numpy as np

from .base import Loss


class CrossEntropyLoss(Loss):
    """
    Categorical Cross-Entropy Loss.
    Debe usarse con Softmax como activación de la última capa.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        L = -(1/m) · Σ y · log(ŷ)

        Args:
            y_pred: salida del softmax, shape (n_classes, m_samples)
            y_true: one-hot labels, shape (n_classes, m_samples)
        """
        m = y_true.shape[1]
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(1.0 / m) * np.sum(y_true * np.log(y_pred_clipped))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Gradiente combinado Softmax + Cross-Entropy:
          dZ = (1/m) · (ŷ - y)
        """
        m = y_true.shape[1]
        return (1.0 / m) * (y_pred - y_true)
