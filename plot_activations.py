"""
plot_activations.py
Genera un gráfico con las curvas de ReLU, Tanh y Softmax para el Slide 3.
Uso: python plot_activations.py [--out results/activations.png]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # backend sin GUI — evita el UserWarning
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def tanh_fn(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def plot_activations(out_path: str) -> None:
    z = np.linspace(-4, 4, 400)

    # ── colores ──
    C_RELU   = "#e74c3c"   # rojo
    C_TANH   = "#3498db"   # azul
    C_SOFT   = "#2ecc71"   # verde
    C_DERIV  = "#aaaaaa"   # gris para derivadas

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ─── Panel 1: ReLU ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#16213e")

    ax1.plot(z, relu(z),       color=C_RELU,  linewidth=2.5, label="ReLU  σ(z) = max(0, z)")
    ax1.plot(z, relu_prime(z), color=C_DERIV, linewidth=1.5, linestyle="--", label="σ'(z)")

    ax1.axhline(0, color="#555", linewidth=0.8)
    ax1.axvline(0, color="#555", linewidth=0.8)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_title("ReLU — capas ocultas", color="white", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("z", color="#aaa", fontsize=11)
    ax1.set_ylabel("σ(z)", color="#aaa", fontsize=11)
    ax1.tick_params(colors="#aaa")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    ax1.legend(fontsize=9, facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    # Anotación del punto de quiebre
    ax1.annotate("Gradiente = 0\npara z ≤ 0",
                 xy=(-1.5, 0.05), fontsize=8, color="#e74c3c",
                 ha="center")
    ax1.annotate("Gradiente = 1\npara z > 0",
                 xy=(2.2, 2.8), fontsize=8, color="#aaa",
                 ha="center")

    # ─── Panel 2: Tanh ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#16213e")

    ax2.plot(z, tanh_fn(z),    color=C_TANH,  linewidth=2.5, label="Tanh  σ(z) = tanh(z)")
    ax2.plot(z, tanh_prime(z), color=C_DERIV, linewidth=1.5, linestyle="--", label="σ'(z) = 1 − tanh²(z)")

    ax2.axhline(0,  color="#555", linewidth=0.8)
    ax2.axhline(1,  color="#555", linewidth=0.5, linestyle=":")
    ax2.axhline(-1, color="#555", linewidth=0.5, linestyle=":")
    ax2.axvline(0,  color="#555", linewidth=0.8)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_title("Tanh — alternativa centrada en 0", color="white", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("z", color="#aaa", fontsize=11)
    ax2.set_ylabel("σ(z)", color="#aaa", fontsize=11)
    ax2.tick_params(colors="#aaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.legend(fontsize=9, facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    ax2.annotate("Salida en (−1, 1)\ncentrada en 0",
                 xy=(2.5, -0.55), fontsize=8, color="#3498db", ha="center")

    # ─── Panel 3: Softmax ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#16213e")

    # Ejemplo con logits para los 10 dígitos
    np.random.seed(7)
    logits_raw = np.array([0.5, -1.2, 2.1, 0.3, -0.4, 0.9, 1.1, -0.7, 0.2, 0.6])
    probs = softmax(logits_raw)
    classes = np.arange(10)
    colors_bar = [C_SOFT if i == np.argmax(probs) else "#1a7a4a" for i in classes]

    bars = ax3.bar(classes, probs, color=colors_bar, edgecolor="#0d5c35", linewidth=0.8, zorder=3)

    # Etiqueta de probabilidad sobre cada barra
    for bar, p in zip(bars, probs):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{p:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color="white")

    ax3.set_xticks(classes)
    ax3.set_xticklabels([str(c) for c in classes])
    ax3.tick_params(colors="#aaa")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")

    ax3.set_title("Softmax — capa de salida", color="white", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Clase (dígito)", color="#aaa", fontsize=11)
    ax3.set_ylabel("Probabilidad", color="#aaa", fontsize=11)
    ax3.set_ylim(0, 0.55)
    ax3.grid(axis="y", alpha=0.2, zorder=0)

    # Fórmula
    ax3.text(0.5, 0.93,
             r"$\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$   →   $\sum_i p_i = 1$",
             transform=ax3.transAxes, ha="center", va="top",
             fontsize=10, color="white",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", alpha=0.8))

    # Leyenda
    from matplotlib.patches import Patch
    ax3.legend(handles=[
        Patch(facecolor=C_SOFT, label=f"Clase predicha: {np.argmax(probs)}  ({max(probs):.0%})"),
        Patch(facecolor="#1a7a4a", label="Otras clases"),
    ], fontsize=9, facecolor="#0f3460", labelcolor="white", framealpha=0.8, loc="upper left")

    # ─── Título global ───────────────────────────────────────────────
    fig.suptitle("Funciones de Activación", color="white",
                 fontsize=15, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/activations.png")
    args = parser.parse_args()
    plot_activations(args.out)


if __name__ == "__main__":
    main()
