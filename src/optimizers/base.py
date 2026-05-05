"""Clase base para optimizadores."""

from typing import List
import numpy as np

from src.layers.base import Layer


class Optimizer:
    """Interfaz: dado un modelo, actualiza sus parámetros usando los gradientes."""

    def step(self, layers: List[Layer]) -> None:
        """Actualiza los parámetros de todas las capas."""
        raise NotImplementedError
