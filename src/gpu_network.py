"""
GpuNetwork — red neuronal que usa cuda_ops.MlpContext para forward/backward
completos en GPU, minimizando transferencias CPU↔GPU.

Softmax y loss siguen en Python. Solo se transfiere:
  - Forward:  X (input) → GPU → logits → CPU → softmax (Python)
  - Backward: grad → GPU → dW,db → CPU → optimizer step (Python)
"""

import numpy as np
from typing import List, Optional


def _load_cuda_ops():
    try:
        import cuda_ops
        return cuda_ops
    except ImportError:
        raise ImportError(
            "cuda_ops no encontrado. Compilá con: "
            "cd 'src/cuda ops' && python setup.py install"
        )


class GpuNetwork:
    """
    MLP que ejecuta forward y backward completos en GPU.
    Compatible con la interfaz de NeuralNetwork (forward, backward, parameters, gradients).
    """

    def __init__(self, architecture: List[int], activation: str = "relu",
                 output_activation: str = "softmax",
                 backend: str = "cuda",
                 dropout: float = 0.0):
        """
        Args:
            architecture: ej [784, 512, 256, 10]
            activation: "relu" o "tanh" para capas ocultas
            output_activation: "softmax" (se aplica en Python) o "none"
            backend: "cuda" o "tensor"
            dropout: probabilidad de dropout (WARNING: no implementado en GPU, se ignora)
        """
        if dropout > 0.0:
            import warnings
            warnings.warn(
                f"Dropout ({dropout}) no está implementado en GpuNetwork. "
                "Se ignorará. Usá backend='cpu' si necesitás dropout.",
                stacklevel=2
            )

        self.architecture = architecture
        self.activation = activation
        self.output_activation = output_activation
        self.backend = backend

        cuda_ops = _load_cuda_ops()
        use_tensor = (backend == "tensor")
        self.ctx = cuda_ops.MlpContext(use_tensor)

        # Inicializar pesos en float32 (He initialization)
        self.weights = []  # list of [W, b] numpy float32 (mutable for in-place updates)
        for i in range(len(architecture) - 1):
            fan_in = architecture[i]
            fan_out = architecture[i + 1]
            W = (np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)).astype(np.float32)
            b = np.zeros((fan_out, 1), dtype=np.float32)
            self.weights.append([W, b])

        # Activaciones: "relu" para ocultas, "none" para la última (softmax se aplica aparte)
        self.act_types = []
        for i in range(len(architecture) - 1):
            if i < len(architecture) - 2:
                self.act_types.append(activation)  # "relu" or "tanh"
            else:
                self.act_types.append("none")  # última capa: sin activación en GPU

        # Gradientes (se llenan después de backward)
        self._grads = None  # list of (dW, db)

        # Cache de output para loss.backward
        self._last_softmax_output = None

    def forward(self, X: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Forward completo: GPU matmul+relu → CPU softmax.

        Args:
            X: (input_size, batch_size) numpy array
        Returns:
            (output_size, batch_size) probabilities after softmax
        """
        # Convertir a float32 C-contiguous
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)

        # Preparar weights como list of tuples
        w_list = [(np.ascontiguousarray(W, dtype=np.float32),
                    np.ascontiguousarray(b, dtype=np.float32))
                   for W, b in self.weights]

        # Forward en GPU → devuelve logits
        logits = self.ctx.forward(X_f32, w_list, self.act_types)

        # Softmax en Python (estabilidad numérica)
        if self.output_activation == "softmax":
            logits_64 = logits.astype(np.float64)
            z_shifted = logits_64 - np.max(logits_64, axis=0, keepdims=True)
            exp_z = np.exp(z_shifted)
            self._last_softmax_output = exp_z / np.sum(exp_z, axis=0, keepdims=True)
            return self._last_softmax_output
        else:
            return logits.astype(np.float64)

    def backward(self, grad: np.ndarray) -> None:
        """
        Backward completo en GPU.

        Args:
            grad: (output_size, batch_size) — gradiente de la loss
                  Para Softmax+CE: grad = (1/m) * (ŷ - y)
        """
        # Softmax backward es pass-through (ya incluido en el grad de CE)
        grad_f32 = np.ascontiguousarray(grad, dtype=np.float32)

        w_list = [(np.ascontiguousarray(W, dtype=np.float32),
                    np.ascontiguousarray(b, dtype=np.float32))
                   for W, b in self.weights]

        # Backward en GPU
        grads = self.ctx.backward(grad_f32, w_list)
        self._grads = grads  # list of (dW, db) tuples

    def parameters(self) -> List[List[np.ndarray]]:
        """
        Devuelve parámetros en el formato que el optimizer espera:
        [[W1, b1], [W2, b2], ...]
        """
        return [[W, b] for W, b in self.weights]

    def gradients(self) -> List[List[np.ndarray]]:
        """
        Devuelve gradientes en el mismo formato que parameters():
        [[dW1, db1], [dW2, db2], ...]
        """
        if self._grads is None:
            # No se ha hecho backward aún — devolver zeros
            return [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.weights]
        return [[dW.astype(np.float64), db.astype(np.float64)]
                for dW, db in self._grads]

    def update_params(self, learning_rate: float) -> None:
        """SGD vanilla (fallback si no se usa optimizer externo)."""
        if self._grads is None:
            return
        for i, (dW, db) in enumerate(self._grads):
            self.weights[i][0] -= learning_rate * dW
            self.weights[i][1] -= learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward sin estado (para inferencia)."""
        return self.forward(X, is_training=False)

    def save(self, path: str, epoch: int = 0) -> None:
        """Guarda pesos en .npz"""
        params = {"epoch": np.array(epoch)}
        for i in range(len(self.weights)):
            W, b = self.weights[i]
            # Guardar indices que coincidan con NeuralNetwork.save
            # LinearLayer está en posiciones 0, 2, 4... (intercalado con activaciones)
            idx = i * 2  # cada linear layer + activation = 2 slots
            params[f"W_{idx}"] = W
            params[f"b_{idx}"] = b
        np.savez(path, **params)

    def load(self, path: str) -> int:
        """Carga pesos desde .npz"""
        data = np.load(path)
        epoch = int(data["epoch"]) if "epoch" in data else 0
        for i in range(len(self.weights)):
            idx = i * 2
            W = data[f"W_{idx}"].astype(np.float32)
            b = data[f"b_{idx}"].astype(np.float32)
            self.weights[i] = (W, b)
        return epoch

    @property
    def layers(self):
        """
        Compatibility shim: devuelve una lista de objetos que tienen
        parameters() y gradients() para que el Optimizer pueda iterar.
        """
        return [_GpuLayerProxy(self, i) for i in range(len(self.weights))]


class _GpuLayerProxy:
    """
    Proxy que simula una Layer con parameters()/gradients() para compatibilidad
    con el Optimizer (que itera sobre network.layers).
    """

    def __init__(self, net: GpuNetwork, idx: int):
        self._net = net
        self._idx = idx

    def parameters(self) -> List[np.ndarray]:
        W, b = self._net.weights[self._idx]
        return [W, b]

    def gradients(self) -> List[np.ndarray]:
        if self._net._grads is None:
            W, b = self._net.weights[self._idx]
            return [np.zeros_like(W), np.zeros_like(b)]
        dW, db = self._net._grads[self._idx]
        return [dW, db]

    def update_params(self, learning_rate: float) -> None:
        pass  # Handled by GpuNetwork.update_params or optimizer
