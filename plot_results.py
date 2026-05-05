"""
Genera gráficos a partir de los CSVs de resultados.
Uso: python plot_results.py [--results-dir results/]
"""

import os
import glob
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # backend sin GUI — evita UserWarning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    plt.close()


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
    plt.close()


def _load_results(csv_path: str) -> dict:
    """Lee el bloque 'results' del _config.json correspondiente al CSV."""
    stem = os.path.splitext(csv_path)[0]
    config_path = stem + "_config.json"
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        data = json.load(f)
    return data.get("results", {})


def compare_by_group(results_dir: str):
    """Agrupa experimentos por tipo y compara val_acc con test_acc en la leyenda."""
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))

    groups = {
        "Learning Rate": [],
        "Arquitectura":  [],
        "Optimizador":   [],
    }

    for f in csv_files:
        name = os.path.basename(f)
        if "lr_" in name:
            groups["Learning Rate"].append(f)
        elif "arch_" in name:
            groups["Arquitectura"].append(f)
        elif "opt_" in name or ("sgd" in name and "arch" not in name):
            groups["Optimizador"].append(f)

    BG = "#1a1a2e"
    PANEL = "#16213e"
    GRID  = "#2c2c4e"
    TEXT  = "#ecf0f1"
    SUB   = "#95a5a6"

    for group_name, files in groups.items():
        if not files:
            continue

        fig, (ax, ax_table) = plt.subplots(
            2, 1, figsize=(11, 7),
            gridspec_kw={"height_ratios": [4, 1]},
        )
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        ax_table.set_facecolor(BG)
        ax_table.axis("off")

        table_rows = []

        for csv_path in files:
            name   = os.path.splitext(os.path.basename(csv_path))[0]
            df     = pd.read_csv(csv_path)
            res    = _load_results(csv_path)

            val_acc  = res.get("val_acc")
            test_acc = res.get("test_acc")

            # Etiqueta de leyenda con resultados
            label = name
            if val_acc is not None:
                label += f"  |  val {val_acc*100:.1f}%"
            if test_acc is not None:
                label += f"  |  test {test_acc*100:.1f}%"

            ax.plot(df["epoch"], df["val_acc"] * 100,
                    label=label, linewidth=2, alpha=0.9)

            table_rows.append([
                name,
                f"{val_acc*100:.2f}%" if val_acc  is not None else "—",
                f"{test_acc*100:.2f}%" if test_acc is not None else "—",
                f"{res.get('val_f1_mean', 0)*100:.2f}%" if res.get('val_f1_mean') else "—",
                f"{res.get('test_f1_mean', 0)*100:.2f}%" if res.get('test_f1_mean') else "—",
            ])

        ax.set_xlabel("Época", color=SUB, fontsize=11)
        ax.set_ylabel("Validation Accuracy (%)", color=SUB, fontsize=11)
        ax.set_title(f"Comparación: {group_name}", color=TEXT,
                     fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=SUB)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2c2c4e")
        ax.legend(fontsize=8.5, facecolor="#0f3460", labelcolor=TEXT,
                  framealpha=0.9, loc="lower right")

        # Tabla resumen debajo del gráfico
        if table_rows:
            col_labels = ["Experimento", "Val acc", "Test acc", "Val F1", "Test F1"]
            tbl = ax_table.table(
                cellText=table_rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.4)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_facecolor("#0f3460" if r == 0 else "#16213e")
                cell.set_text_props(color=TEXT)
                cell.set_edgecolor("#2c2c4e")

        plt.tight_layout()
        out = os.path.join(results_dir,
                           f"compare_{group_name.lower().replace(' ', '_')}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"Guardado: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    plot_loss_curves(args.results_dir)
    compare_by_group(args.results_dir)
