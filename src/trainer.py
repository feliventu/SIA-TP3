"""
Trainer: loop de entrenamiento con mini-batch, logging a CSV, y early stopping.
"""

import csv
import os
import time
from typing import List, Dict, Optional

import numpy as np

from src.network import NeuralNetwork
from src.losses.base import Loss
from src.optimizers.base import Optimizer


class Trainer:
    """Entrena una NeuralNetwork y loguea métricas a CSV."""

    def __init__(self, network: NeuralNetwork, loss_fn: Loss,
                 learning_rate: float = 0.01, batch_size: int = 32,
                 seed: Optional[int] = None,
                 optimizer: Optional[Optimizer] = None):
        self.network = network
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer

        if seed is not None:
            np.random.seed(seed)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              log_path: Optional[str] = None, print_every: int = 100,
              early_stopping_patience: Optional[int] = None,
              initial_epoch: int = 0) -> List[dict]:
        """
        Loop de entrenamiento principal.

        Args:
            X_train: (n_features, m_samples)
            y_train: (n_output, m_samples)
            epochs: cantidad de épocas
            X_val, y_val: datos de validación (opcional)
            log_path: ruta al CSV donde guardar métricas
            print_every: cada cuántas épocas imprimir
            early_stopping_patience: épocas sin mejora para parar (None = no parar)

        Returns:
            Lista de dicts con métricas por época
        """
        history = []
        best_val_loss = float("inf")
        patience_counter = 0
        m = X_train.shape[1]

        csv_file = None
        csv_writer = None
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            mode = "a" if initial_epoch > 0 and os.path.exists(log_path) else "w"
            csv_file = open(log_path, mode, newline="")
            fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if mode == "w":
                csv_writer.writeheader()

        try:
            for epoch in range(initial_epoch + 1, initial_epoch + epochs + 1):
                t0 = time.time()

                # --- Shuffle y mini-batches ---
                indices = np.random.permutation(m)
                X_shuffled = X_train[:, indices]
                y_shuffled = y_train[:, indices]

                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, m, self.batch_size):
                    end = min(start + self.batch_size, m)
                    X_batch = X_shuffled[:, start:end]
                    y_batch = y_shuffled[:, start:end]

                    # Forward
                    y_pred = self.network.forward(X_batch)

                    # Loss
                    batch_loss = self.loss_fn.forward(y_pred, y_batch)
                    epoch_loss += batch_loss

                    # Backward
                    grad = self.loss_fn.backward(y_pred, y_batch)
                    self.network.backward(grad)

                    # Update: usar optimizer si hay, sino SGD vanilla
                    if self.optimizer is not None:
                        self.optimizer.step(self.network.layers)
                    else:
                        self.network.update_params(self.lr)

                    n_batches += 1

                epoch_loss /= n_batches
                elapsed = time.time() - t0

                # --- Métricas ---
                row = {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "train_acc": self._accuracy(X_train, y_train),
                    "val_loss": None,
                    "val_acc": None,
                    "time_s": elapsed,
                }

                if X_val is not None and y_val is not None:
                    val_pred = self.network.forward(X_val)
                    row["val_loss"] = self.loss_fn.forward(val_pred, y_val)
                    row["val_acc"] = self._accuracy(X_val, y_val)

                history.append(row)

                # CSV
                if csv_writer is not None:
                    csv_writer.writerow(row)
                    csv_file.flush()

                # Print
                if epoch % print_every == 0 or epoch == 1:
                    msg = f"Epoch {epoch}/{epochs} | loss: {epoch_loss:.6f} | acc: {row['train_acc']:.4f}"
                    if row["val_loss"] is not None:
                        msg += f" | val_loss: {row['val_loss']:.6f} | val_acc: {row['val_acc']:.4f}"
                    msg += f" | {elapsed:.3f}s"
                    print(msg)

                # Early stopping
                if early_stopping_patience is not None and X_val is not None:
                    val_loss = row["val_loss"]
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping en epoch {epoch}")
                            break

        finally:
            if csv_file is not None:
                csv_file.close()

        return history

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula accuracy. Funciona para:
        - Clasificación multiclase (one-hot): argmax de predicción vs argmax de label
        - Binario/regresión (1 salida): redondea predicción y compara
        """
        y_pred = self.network.forward(X)

        if y.shape[0] > 1:
            # Multiclase: comparar argmax
            pred_labels = np.argmax(y_pred, axis=0)
            true_labels = np.argmax(y, axis=0)
        else:
            # Binario/escalar: redondear
            pred_labels = np.round(y_pred).flatten()
            true_labels = y.flatten()

        return np.mean(pred_labels == true_labels)
