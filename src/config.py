"""Carga configuración desde YAML y construye los objetos necesarios."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import yaml

from src.network import NeuralNetwork
from src.losses.mse import MSELoss
from src.losses.cross_entropy import CrossEntropyLoss
from src.optimizers.sgd import SGD
from src.optimizers.adam import Adam
from src.optimizers.base import Optimizer
from src.losses.base import Loss


LOSSES = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
}

OPTIMIZERS = {
    "sgd": lambda cfg: SGD(learning_rate=cfg.get("learning_rate", 0.01),
                           momentum=0.0),
    "sgd_momentum": lambda cfg: SGD(learning_rate=cfg.get("learning_rate", 0.01),
                                     momentum=cfg.get("momentum", 0.9)),
    "adam": lambda cfg: Adam(learning_rate=cfg.get("learning_rate", 0.001),
                             beta1=cfg.get("beta1", 0.9),
                             beta2=cfg.get("beta2", 0.999)),
}


@dataclass
class ExperimentConfig:
    """Configuración completa de un experimento."""
    experiment_name: str = "default"
    architecture: list = field(default_factory=lambda: [784, 128, 10])
    activation: str = "relu"
    output_activation: str = "softmax"
    loss: str = "cross_entropy"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    early_stopping_patience: Optional[int] = None
    train_val_split: float = 0.8
    seed: int = 42
    backend: str = "cpu"  # "cpu", "cuda", "tensor"

    def build_network(self) -> NeuralNetwork:
        return NeuralNetwork.from_config(
            architecture=self.architecture,
            activation=self.activation,
            output_activation=self.output_activation,
            backend=self.backend,
        )

    def build_loss(self) -> Loss:
        return LOSSES[self.loss]()

    def build_optimizer(self) -> Optimizer:
        cfg = {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "beta1": self.beta1,
            "beta2": self.beta2,
        }
        return OPTIMIZERS[self.optimizer](cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Para guardar junto con resultados."""
        return {
            "experiment_name": self.experiment_name,
            "architecture": self.architecture,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "momentum": self.momentum,
            "seed": self.seed,
            "backend": self.backend,
        }


def load_config(path: str) -> ExperimentConfig:
    """Carga un ExperimentConfig desde un archivo YAML."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


def load_configs(path: str) -> list:
    """Carga múltiples configs desde YAML."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        return [ExperimentConfig(**exp) for exp in data]
    elif isinstance(data, dict) and "experiments" in data:
        return [ExperimentConfig(**exp) for exp in data["experiments"]]
    elif isinstance(data, dict):
        return [ExperimentConfig(**data)]
    else:
        raise ValueError("Formato de config.yaml no reconocido.")
