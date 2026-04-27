"""
plot_single.py
Grafica el historial de entrenamiento de UN experimento: loss, accuracy y tiempo por época.
Útil para el slide de configuración óptima y para comparar tiempos CPU vs GPU.

Uso:
    python plot_single.py --name digit_lr_0001_cpu
    python plot_single.py --csv results/digit_lr_0001_cpu.csv
    python plot_single.py --name digit_lr_0001_cpu --out results/optimal_result.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


# ── Paleta dark ──────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_PANEL  = "#16213e"
C_TRAIN   = "#e74c3c"   # rojo — train
C_VAL     = "#2ecc71"   # verde — val
C_TIME    = "#f39c12"   # naranja — tiempo
C_GRID    = "#2c2c4e"
C_TEXT    = "#ecf0f1"
C_SUBTEXT = "#95a5a6"


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["train_loss", "train_acc", "val_loss", "val_acc", "time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_test_results(csv_path: str) -> dict:
    """Busca el _config.json del experimento y devuelve el bloque 'results' si existe."""
    import json
    stem = os.path.splitext(csv_path)[0]  # quita .csv
    config_path = stem + "_config.json"
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        data = json.load(f)
    return data.get("results", {})


def plot_single(csv_path: str, out_path: str, title: str = None) -> None:
    df = load_csv(csv_path)
    test_results = load_test_results(csv_path)

    has_val  = "val_acc" in df.columns and df["val_acc"].notna().any()
    has_time = "time_s"  in df.columns and df["time_s"].notna().any()
    test_acc = test_results.get("test_acc")      # float o None
    test_f1_mean = test_results.get("test_f1_mean")  # float o None

    n_panels = 3 if has_time else 2
    name = title or os.path.splitext(os.path.basename(csv_path))[0]

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.35)

    # ── Panel 1: Loss ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_PANEL)

    ax1.plot(df["epoch"], df["train_loss"], color=C_TRAIN, linewidth=2,
             label="Train loss", zorder=3)
    if has_val:
        ax1.plot(df["epoch"], df["val_loss"], color=C_VAL, linewidth=2,
                 linestyle="--", label="Val loss", zorder=3)

    # Marcar early stopping si el entrenamiento paró antes del máximo
    max_epoch_possible = df["epoch"].max()
    _style_ax(ax1, "Loss por época", "Época", "Loss")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, facecolor="#0f3460", labelcolor=C_TEXT, framealpha=0.8)

    # ── Panel 2: Accuracy ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG_PANEL)

    ax2.plot(df["epoch"], df["train_acc"] * 100, color=C_TRAIN, linewidth=2,
             label="Train acc", zorder=3)

    if has_val:
        ax2.plot(df["epoch"], df["val_acc"] * 100, color=C_VAL, linewidth=2,
                 linestyle="--", label="Val acc", zorder=3)
        final_val = df["val_acc"].dropna().iloc[-1] * 100
        # Línea de referencia punteada
        ax2.axhline(final_val, color=C_VAL, linewidth=0.8, linestyle=":",
                    alpha=0.5)
        # Etiqueta a la derecha, separada 1.5% por encima de la línea
        ax2.text(df["epoch"].max() * 0.98, final_val + 1.5,
                 f"Val: {final_val:.2f}%",
                 color=C_TEXT, fontsize=9, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f3460",
                           edgecolor=C_VAL, alpha=0.85))

    ax2.set_ylim(
        max(0, df["train_acc"].min() * 100 - 5),
        103
    )

    # Línea de test accuracy (desde el JSON)
    if test_acc is not None:
        test_pct = test_acc * 100
        ax2.axhline(test_pct, color=C_TRAIN, linewidth=1.5, linestyle="--",
                    alpha=0.8, zorder=4)
        label_test = f"Test: {test_pct:.2f}%"
        if test_f1_mean is not None:
            label_test += f"  |  F1: {test_f1_mean:.3f}"
        ax2.text(df["epoch"].max() * 0.98, test_pct - 1.5,
                 label_test,
                 color=C_TEXT, fontsize=9, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f3460",
                           edgecolor=C_TRAIN, alpha=0.85))

    _style_ax(ax2, "Accuracy por época", "Época", "Accuracy (%)")
    ax2.legend(fontsize=9, facecolor="#0f3460", labelcolor=C_TEXT, framealpha=0.8)

    # ── Panel 3: Tiempo por época (opcional) ──────────────────────────────────
    if has_time:
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(BG_PANEL)

        ax3.plot(df["epoch"], df["time_s"], color=C_TIME, linewidth=1.5,
                 alpha=0.7, zorder=3)
        ax3.fill_between(df["epoch"], df["time_s"], alpha=0.15, color=C_TIME)

        mean_t = df["time_s"].mean()
        ax3.axhline(mean_t, color=C_TIME, linestyle="--", linewidth=1.2, alpha=0.9)
        ax3.text(df["epoch"].max() * 0.02, mean_t * 1.03,
                 f"Media: {mean_t:.3f}s/época",
                 color=C_TIME, fontsize=9)

        _style_ax(ax3, "Tiempo por época", "Época", "Tiempo (s)")

    # ── Título global ─────────────────────────────────────────────────────────
    fig.suptitle(name, color=C_TEXT, fontsize=14, fontweight="bold", y=1.02)

    # ── Guardar ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Guardado: {out_path}")

    # Resumen en consola
    _print_summary(df, name)


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=C_SUBTEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=C_SUBTEXT, fontsize=10)
    ax.tick_params(colors=C_SUBTEXT)
    ax.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c2c4e")


def _print_summary(df: pd.DataFrame, name: str):
    print(f"\n── Resumen: {name} ──────────────────────────────────")
    print(f"  Épocas entrenadas : {df['epoch'].max()}")
    print(f"  Train loss final  : {df['train_loss'].iloc[-1]:.6f}")
    print(f"  Train acc final   : {df['train_acc'].iloc[-1]*100:.2f}%")
    if "val_acc" in df.columns and df["val_acc"].notna().any():
        best_val = df["val_acc"].max() * 100
        best_ep  = df.loc[df["val_acc"].idxmax(), "epoch"]
        print(f"  Val acc final     : {df['val_acc'].iloc[-1]*100:.2f}%")
        print(f"  Mejor val acc     : {best_val:.2f}% (epoch {best_ep})")
    if "time_s" in df.columns and df["time_s"].notna().any():
        mean_t = df["time_s"].mean()
        total_t = df["time_s"].sum()
        print(f"  Tiempo medio/época: {mean_t:.3f}s")
        print(f"  Tiempo total      : {total_t:.1f}s ({total_t/60:.1f} min)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Grafica un único experimento")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Nombre del experimento (busca en results/<name>.csv)")
    group.add_argument("--csv",  help="Ruta directa al CSV")
    parser.add_argument("--out",   default=None, help="Ruta de salida del PNG (default: results/<name>_plot.png)")
    parser.add_argument("--title", default=None, help="Título del gráfico (default: nombre del archivo)")
    parser.add_argument("--results-dir", default="results", help="Directorio de resultados")
    args = parser.parse_args()

    if args.name:
        csv_path = os.path.join(args.results_dir, f"{args.name}.csv")
        out_path = args.out or os.path.join(args.results_dir, f"{args.name}_plot.png")
        title    = args.title or args.name
    else:
        csv_path = args.csv
        stem     = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = args.out or os.path.join(os.path.dirname(csv_path), f"{stem}_plot.png")
        title    = args.title or stem

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró {csv_path}")
        return

    plot_single(csv_path, out_path, title=title)


if __name__ == "__main__":
    main()
