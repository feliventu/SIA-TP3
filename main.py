"""
Main: corre todos los experimentos definidos en config.yaml.
Carga datos, entrena, evalúa en test, y guarda resultados a CSV.
"""

import sys
import os
import ast
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from src.config import load_configs, ExperimentConfig
from src.trainer import Trainer
from src.metrics import accuracy, confusion_matrix, f1_per_class


# ============================================================
# Data Loading
# ============================================================

def load_dataset(path: str, n_classes: int = 10):
    """
    Carga dataset. Soporta:
    - .csv (formato digits.csv con columnas 'image' y 'label')
    - .npz (formato KMNIST con arrays 'images' y 'labels')
    
    Retorna X (n_features, m) y y one-hot (n_classes, m).
    """
    if path.endswith(".npz"):
        data = np.load(path)
        images = data["images"]   # (m, 784)
        labels = data["labels"]   # (m,)
    else:
        df = pd.read_csv(path)
        images = np.array([np.array(ast.literal_eval(s), dtype=np.float64)
                           for s in df["image"].values])  # (m, 784)
        labels = df["label"].values.astype(int)  # (m,)
    
    X = images.T  # (n_features, m)
    
    # One-hot encoding
    y = np.zeros((n_classes, len(labels)))
    for i, label in enumerate(labels):
        y[label, i] = 1.0
    
    return X, y


def train_val_split(X, y, split: float = 0.8, seed: int = 42):
    """Divide datos en train/val."""
    rng = np.random.RandomState(seed)
    m = X.shape[1]
    indices = rng.permutation(m)
    split_idx = int(m * split)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return X[:, train_idx], y[:, train_idx], X[:, val_idx], y[:, val_idx]


# ============================================================
# Run Experiments
# ============================================================

def run_experiment(config: ExperimentConfig, X_train, y_train, X_val, y_val,
                   X_test=None, y_test=None):
    """Ejecuta un experimento y devuelve resultados."""
    
    print(f"\n{'='*60}")
    print(f"Experimento: {config.experiment_name}")
    print(f"  Arquitectura: {config.architecture}")
    print(f"  Activación: {config.activation} | Salida: {config.output_activation}")
    print(f"  Optimizador: {config.optimizer} | LR: {config.learning_rate}")
    print(f"  Batch: {config.batch_size} | Epochs: {config.epochs}")
    print(f"{'='*60}")
    
    # Construir modelo
    network = config.build_network()
    loss_fn = config.build_loss()
    optimizer = config.build_optimizer()
    
    initial_epoch = 0
    model_path = f"results/{config.experiment_name}_model.npz"
    if os.path.exists(model_path):
        print(f"Cargando pesos existentes desde {model_path} (continuando entrenamiento)...")
        initial_epoch = network.load(model_path)
    else:
        print("Empezando a entrenar desde cero (no hay archivo guardado anterior)...")
    
    # Trainer
    log_path = f"results/{config.experiment_name}.csv"
    trainer = Trainer(
        network, loss_fn,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        seed=config.seed,
        optimizer=optimizer,
        augmentation=config.augmentation,
        backend=config.backend,
    )
    
    # Entrenar
    history = trainer.train(
        X_train, y_train, config.epochs,
        X_val=X_val, y_val=y_val,
        log_path=log_path,
        print_every=5,
        early_stopping_patience=config.early_stopping_patience,
        initial_epoch=initial_epoch,
    )
    
    # Evaluación final en validación
    val_pred = network.forward(X_val)
    val_acc = accuracy(val_pred, y_val)
    cm = confusion_matrix(val_pred, y_val)
    f1 = f1_per_class(cm)
    
    print(f"\n--- Resultados finales ---")
    print(f"Val accuracy: {val_acc:.4f}")
    print(f"F1 por clase: {np.round(f1, 3)}")
    print(f"F1 promedio:  {np.mean(f1):.4f}")
    
    # Test set si existe
    test_acc = None
    test_f1  = None
    if X_test is not None and y_test is not None:
        test_pred = network.forward(X_test)
        test_acc  = float(accuracy(test_pred, y_test))
        test_cm   = confusion_matrix(test_pred, y_test)
        test_f1   = f1_per_class(test_cm).tolist()
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test F1 promedio: {np.mean(test_f1):.4f}")
    
    # Guardar config + resultados junto al CSV
    config_path = f"results/{config.experiment_name}_config.json"
    save_data = config.to_dict()
    save_data["results"] = {
        "val_acc":      float(val_acc),
        "val_f1":       f1.tolist(),
        "val_f1_mean":  float(np.mean(f1)),
        "test_acc":     test_acc,
        "test_f1":      test_f1,
        "test_f1_mean": float(np.mean(test_f1)) if test_f1 is not None else None,
        "epochs_trained": initial_epoch + len(history),
    }
    with open(config_path, "w") as f:
        json.dump(save_data, f, indent=2)
    
    # Guardar modelo
    final_epoch = initial_epoch + len(history)
    network.save(f"results/{config.experiment_name}_model.npz", epoch=final_epoch)
    
    return history, val_acc


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MLP Benchmark Runner")
    parser.add_argument("--config", default="config.yaml", help="Path al YAML de config")
    parser.add_argument("--data", default="data/digits.csv", help="Dataset de entrenamiento")
    parser.add_argument("--test", default="test_data/digits_test.csv", help="Dataset de test")
    parser.add_argument("--test-only", action="store_true", help="Solo evaluar modelo guardado, no entrenar")
    args = parser.parse_args()
    
    # Cargar datos
    print("Cargando datos de entrenamiento...")
    X, y = load_dataset(args.data)
    print(f"  Datos: {X.shape[1]} muestras, {X.shape[0]} features")
    
    # Test set
    X_test, y_test = None, None
    if os.path.exists(args.test):
        print("Cargando datos de test...")
        X_test, y_test = load_dataset(args.test)
        print(f"  Test: {X_test.shape[1]} muestras")
    
    # Cargar configs
    configs = load_configs(args.config)
    
    os.makedirs("results", exist_ok=True)
    
    # --test-only: cargar modelo y evaluar sin entrenar
    if args.test_only:
        print("\n--- MODO TEST-ONLY: evaluando modelos guardados ---\n")
        for config in configs:
            model_path = f"results/{config.experiment_name}_model.npz"
            if not os.path.exists(model_path):
                print(f"No se encontró modelo para '{config.experiment_name}', saltando...")
                continue
            
            network = config.build_network()
            network.load(model_path)
            
            print(f"Modelo: {config.experiment_name}")
            
            if X_test is not None and y_test is not None:
                test_pred = network.forward(X_test)
                test_acc = accuracy(test_pred, y_test)
                cm = confusion_matrix(test_pred, y_test)
                f1 = f1_per_class(cm)
                print(f"  Test accuracy: {test_acc:.4f}")
                print(f"  F1 por clase:  {np.round(f1, 3)}")
                print(f"  F1 promedio:   {np.mean(f1):.4f}")
                print(f"  Confusion matrix:\n{cm}\n")
            else:
                print("  No hay datos de test disponibles.")
        return
    
    # Modo normal: entrenar
    print(f"\n{len(configs)} experimentos a correr\n")
    
    results_summary = []
    for config in configs:
        X_train, y_train, X_val, y_val = train_val_split(
            X, y, split=config.train_val_split, seed=config.seed
        )
        
        history, val_acc = run_experiment(
            config, X_train, y_train, X_val, y_val,
            X_test=X_test, y_test=y_test
        )
        
        results_summary.append({
            "experiment": config.experiment_name,
            "val_acc": val_acc,
            "epochs": len(history),
            "final_loss": history[-1]["train_loss"],
        })
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE TODOS LOS EXPERIMENTOS")
    print(f"{'='*60}")
    for r in sorted(results_summary, key=lambda x: x["val_acc"], reverse=True):
        print(f"  {r['experiment']:30s} | val_acc: {r['val_acc']:.4f} | loss: {r['final_loss']:.6f}")


if __name__ == "__main__":
    main()

