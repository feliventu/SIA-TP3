import numpy as np

class Loss:

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calcula el valor de la loss (escalar).

        Args:
            y_pred: predicciones, shape (n_output, m_samples)
            y_true: valores reales, shape (n_output, m_samples)

        Returns:
            float: valor promedio de la loss sobre el batch
        """
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calcula dL/dŷ — el gradiente de la loss respecto a las predicciones.

        Returns:
            gradiente, same shape que y_pred
        """
        raise NotImplementedError
