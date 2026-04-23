"""
Genera gráficos a partir de los CSVs de resultados.
Uso: python plot_results.py [--results-dir results/]
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curves(results_dir: str):
    """Grafica train_loss vs epoch para todos los experimentos."""
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    csv_files = [f for f in csv_files if not f.endswith("_config.json")]

    if not csv_files:
        print("No se encontraron archivos CSV en", results_dir)
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)

        axes[0].plot(df["epoch"], df["train_loss"], label=name, alpha=0.8)
        axes[1].plot(df["epoch"], df["train_acc"], label=name, alpha=0.8)

        if "val_loss" in df.columns and df["val_loss"].notna().any():
            axes[0].plot(df["epoch"], df["val_loss"], linestyle="--", alpha=0.5)
        if "val_acc" in df.columns and df["val_acc"].notna().any():
            axes[1].plot(df["epoch"], df["val_acc"], linestyle="--", alpha=0.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison.png"), dpi=150)
    print(f"Guardado: {results_dir}/comparison.png")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix",
                          save_path: str = None):
    """Grafica una confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")

    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)

    # Anotar valores
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Guardado: {save_path}")
    plt.show()


def compare_by_group(results_dir: str):
    """Agrupa experimentos por tipo (lr, arch, opt, act) y compara."""
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))

    groups = {
        "Learning Rate": [],
        "Arquitectura": [],
        "Optimizador": [],
        "Activación": [],
    }

    for f in csv_files:
        name = os.path.basename(f)
        if "lr_" in name:
            groups["Learning Rate"].append(f)
        elif "arch_" in name:
            groups["Arquitectura"].append(f)
        elif "sgd" in name or "adam" in name:
            groups["Optimizador"].append(f)
        elif "tanh" in name:
            groups["Activación"].append(f)

    for group_name, files in groups.items():
        if not files:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for csv_path in files:
            name = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)
            ax.plot(df["epoch"], df["val_acc"], label=name, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"Comparación: {group_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"compare_{group_name.lower().replace(' ', '_')}.png"), dpi=150)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    plot_loss_curves(args.results_dir)
    compare_by_group(args.results_dir)
