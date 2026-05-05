"""
plot_digit_frequency.py
Analiza la distribución de clases en digits.csv y genera un gráfico de barras.
Resalta en rojo la clase 8 (ausente o sub-representada) vs el resto en azul.

Uso:
    python plot_digit_frequency.py [--data data/digits.csv] [--out results/digit_frequency.png]
"""

import argparse
import ast
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def load_labels(path: str) -> np.ndarray:
    """Carga solo las etiquetas del CSV (columna 'label')."""
    if path.endswith(".npz"):
        data = np.load(path)
        return data["labels"].astype(int)
    else:
        df = pd.read_csv(path)
        return df["label"].values.astype(int)


def plot_frequency(labels: np.ndarray, out_path: str) -> None:
    classes = np.arange(10)
    counts = np.array([(labels == c).sum() for c in classes])
    total = len(labels)

    # Colores: rojo para clases ausentes o con mucha menos frecuencia que la media
    mean_count = counts[counts > 0].mean()
    threshold = mean_count * 0.3          # menos del 30% de la media → alerta
    colors = []
    for c in counts:
        if c == 0:
            colors.append("#e74c3c")      # rojo brillante — completamente ausente
        elif c < threshold:
            colors.append("#e67e22")      # naranja — sub-representado
        else:
            colors.append("#3498db")      # azul — normal

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    # Etiqueta con conteo y % sobre cada barra
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        label = f"{count:,}\n({pct:.1f}%)" if count > 0 else "AUSENTE"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.002,
            label,
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#2c3e50",
        )

    # Línea de media
    ax.axhline(mean_count, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Media por clase: {mean_count:,.0f}", zorder=2)

    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="white", label="Presente (normal)"),
        Patch(facecolor="#e67e22", edgecolor="white", label="Sub-representado (<30% media)"),
        Patch(facecolor="#e74c3c", edgecolor="white", label="Ausente en train"),
    ]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color="#2ecc71",
              linestyle="--", linewidth=1.5, label=f"Media: {mean_count:,.0f}")],
              loc="upper right", fontsize=9)

    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes], fontsize=11)
    ax.set_xlabel("Dígito", fontsize=12)
    ax.set_ylabel("Cantidad de muestras", fontsize=12)
    ax.set_title("Distribución de clases en digits.csv (training set)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Guardado: {out_path}")

    # Resumen en consola
    print("\n--- Distribución de clases ---")
    print(f"{'Clase':>6} {'Cantidad':>10} {'%':>7}")
    print("-" * 28)
    for c in classes:
        pct = counts[c] / total * 100
        flag = " ← AUSENTE" if counts[c] == 0 else (" ← bajo" if counts[c] < threshold else "")
        print(f"  {c:>4}   {counts[c]:>9,}   {pct:>6.2f}%{flag}")
    print(f"\n  Total: {total:,} muestras")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Frecuencia de dígitos en el dataset")
    parser.add_argument("--data", default="data/digits.csv", help="Ruta al CSV de entrenamiento")
    parser.add_argument("--out", default="results/digit_frequency.png", help="Ruta de salida del gráfico")
    args = parser.parse_args()

    print(f"Cargando {args.data} ...")
    labels = load_labels(args.data)
    print(f"  {len(labels):,} muestras encontradas")

    plot_frequency(labels, args.out)


if __name__ == "__main__":
    main()
