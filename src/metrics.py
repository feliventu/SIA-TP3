import numpy as np
from typing import Optional


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Accuracy para one-hot o escalar."""
    if y_true.shape[0] > 1:
        pred_labels = np.argmax(y_pred, axis=0)
        true_labels = np.argmax(y_true, axis=0)
    else:
        pred_labels = np.round(y_pred).flatten()
        true_labels = y_true.flatten()
    return np.mean(pred_labels == true_labels)


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray,
                     n_classes: int = 10) -> np.ndarray:
    """Matriz de confusión (n_classes x n_classes)."""
    pred_labels = np.argmax(y_pred, axis=0)
    true_labels = np.argmax(y_true, axis=0)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm


def precision_per_class(cm: np.ndarray) -> np.ndarray:
    """Precisión por clase desde la confusion matrix."""
    col_sums = cm.sum(axis=0)
    col_sums[col_sums == 0] = 1  # evitar div/0
    return np.diag(cm) / col_sums


def recall_per_class(cm: np.ndarray) -> np.ndarray:
    """Recall por clase desde la confusion matrix."""
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1  # evitar div/0
    return np.diag(cm) / row_sums


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    """F1 por clase."""
    p = precision_per_class(cm)
    r = recall_per_class(cm)
    denom = p + r
    denom[denom == 0] = 1  # evitar div/0
    return 2 * p * r / denom
