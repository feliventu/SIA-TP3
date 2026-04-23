"""
Test de validación: XOR con MLP.

XOR no es linealmente separable -> necesita al menos una capa oculta.
Probamos arquitecturas [2,2,1] y [2,3,2,1] con Tanh + MSE.
"""

import sys
import os

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.network import NeuralNetwork
from src.losses.mse import MSELoss
from src.trainer import Trainer
from typing import List

def test_xor(architecture: List[int], lr: float = 0.1, epochs: int = 5000,
             seed: int = 42):
    """Entrena un MLP para resolver XOR y reporta resultados."""

    print(f"\n{'='*50}")
    print(f"XOR con arquitectura {architecture}")
    print(f"{'='*50}")

    # Datos XOR: shape (features, samples) = (2, 4)
    X = np.array([[-1, 1, -1, 1],
                  [1, -1, -1, 1]], dtype=np.float64)
    y = np.array([[1, 1, -1, -1]], dtype=np.float64)  # shape (1, 4)

    # Crear red con Tanh (salidas en [-1,1])
    network = NeuralNetwork.from_config(
        architecture=architecture,
        activation="tanh",
        # output_activation="same" por defecto -> Tanh en la salida
    )

    # Entrenar
    loss_fn = MSELoss()
    trainer = Trainer(network, loss_fn, learning_rate=lr,
                      batch_size=4, seed=seed)  # batch_size=4 = batch completo
    
    history = trainer.train(
        X, y, epochs=epochs,
        log_path=f"results/xor_{'_'.join(map(str, architecture))}.csv",
        print_every=1000,
    )

    # Resultado final
    y_pred = network.predict(X)
    print(f"\nPredicciones finales:")
    for i in range(4):
        x1, x2 = X[0, i], X[1, i]
        pred = y_pred[0, i]
        expected = y[0, i]
        print(f"  ({x1:+.0f}, {x2:+.0f}) -> pred: {pred:+.4f}  expected: {expected:+.0f}")

    final_loss = history[-1]["train_loss"]
    print(f"\nLoss final: {final_loss:.6f}")
    print(f"Convergió: {'✓' if final_loss < 0.01 else '✗'}")


if __name__ == "__main__":
    # Probar ambas arquitecturas
    test_xor([2, 2, 1], lr=0.1, epochs=5000)
    test_xor([2, 3, 2, 1], lr=0.1, epochs=5000)
