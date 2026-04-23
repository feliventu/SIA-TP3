"""
Mean Squared Error (MSE) Loss.

== Fórmula ==
  L = (1 / 2m) · Σ (ŷ - y)²

  dL/dŷ = (1/m) · (ŷ - y)

"""

import numpy as np

from .base import Loss


class MSELoss(Loss):

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        m = y_true.shape[1]
        return (1.0 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        m = y_true.shape[1]
        return (1.0 / m) * (y_pred - y_true)
