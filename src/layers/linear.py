import numpy as np
from typing import List

from .base import Layer


def _load_cuda_ops():
    """Intenta importar el módulo CUDA compilado."""
    try:
        import cuda_ops
        return cuda_ops
    except ImportError:
        raise ImportError(
            "cuda_ops no encontrado. Compilá con: "
            "cd src/cuda_ops && python setup.py install"
        )


class LinearLayer(Layer):

    def __init__(self, input_size: int, output_size: int, backend: str = "cpu"):
        self.backend = backend
        self._cuda_ops = None

        self.W = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((output_size, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._input = None

        # Cargar módulo CUDA solo si se necesita
        if backend in ("cuda", "tensor"):
            self._cuda_ops = _load_cuda_ops()

    def _matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiplica matrices usando el backend configurado."""
        if self.backend == "cpu":
            return np.dot(A, B)
        else:
            # CUDA necesita arrays contiguos en memoria (C-order, float32)
            # .T en NumPy NO copia datos, solo cambia strides → CUDA lee basura
            A = np.ascontiguousarray(A, dtype=np.float32)
            B = np.ascontiguousarray(B, dtype=np.float32)
            use_tensor = (self.backend == "tensor")
            result = self._cuda_ops.matmul(A, B, use_tensor)
            return result.astype(np.float64)  # volver a float64 para consistencia

    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Z = W · input + b"""
        self._input = input_data
        return self._matmul(self.W, input_data) + self.b

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Calcula dW, db, y dA_prev."""
        # output_gradient ya viene escalado por batch-size desde la loss.
        # Evitamos promediar otra vez para no reducir de más la magnitud del gradiente.
        self.dW = self._matmul(output_gradient, self._input.T)
        self.db = np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = self._matmul(self.W.T, output_gradient)

        return input_gradient

    def parameters(self) -> List[np.ndarray]:
        return [self.W, self.b]

    def gradients(self) -> List[np.ndarray]:
        return [self.dW, self.db]

    def update_params(self, learning_rate: float) -> None:
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db