"""Publication-style figures for the QCBM notebook and README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.bas import bitstring

INK = "#102a43"
TEAL = "#0d9488"
CORAL = "#c23b22"
SLATE = "#486581"
PAPER = "#f8fafc"
VALID_EDGE = "#0f766e"
INVALID_EDGE = "#b91c1c"

PATTERN_CMAP = ListedColormap(["#f8fafc", "#102a43"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PAPER,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 11,
            "axes.titleweight": "semibold",
            "axes.grid": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )


def _draw_pattern(ax, pattern: np.ndarray, n: int, title: str | None = None, edge: str = INK) -> None:
    grid = np.asarray(pattern).reshape(n, n)
    ax.imshow(grid, cmap=PATTERN_CMAP, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(edge)
        spine.set_linewidth(1.8)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#94a3b8", linewidth=0.8)
    if title is not None:
        ax.set_xlabel(title, fontsize=9, color=SLATE)


def save_target_patterns(patterns: np.ndarray, n: int, path: Path) -> Path:
    apply_style()
    n_patterns = len(patterns)
    cols = min(6, n_patterns)
    rows = int(np.ceil(n_patterns / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols + 0.8, 1.9 * rows + 0.6))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n_patterns:
            bits = "".join(str(int(b)) for b in patterns[i])
            _draw_pattern(ax, patterns[i], n, title=bits)
        else:
            ax.axis("off")
    fig.suptitle(f"{n}×{n} Bars-and-Stripes support ({n_patterns} patterns)", color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_loss_curve(losses: list[float], validity: list[float], path: Path) -> Path:
    apply_style()
    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
    steps = np.arange(len(losses))
    ax1.plot(steps, losses, color=CORAL, lw=2.2, label="KL(π ‖ pθ)")
    ax1.set_xlabel("Adam step")
    ax1.set_ylabel("KL divergence", color=CORAL)
    ax1.tick_params(axis="y", colors=CORAL)
    ax2 = ax1.twinx()
    ax2.plot(steps, validity, color=TEAL, lw=2.2, label="P(valid BAS)")
    ax2.set_ylabel("probability mass on BAS", color=TEAL)
    ax2.tick_params(axis="y", colors=TEAL)
    ax2.set_ylim(-0.02, 1.05)
    ax1.set_title("Training on exact 4-qubit Born probabilities")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_probability_comparison(
    target: np.ndarray,
    model: np.ndarray,
    valid_indices: np.ndarray,
    n_qubits: int,
    path: Path,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    xs = np.arange(len(target))
    ax.bar(xs - 0.18, target, width=0.36, color=INK, alpha=0.85, label="target π(x)")
    ax.bar(xs + 0.18, model, width=0.36, color=TEAL, alpha=0.9, label="model pθ(x)")
    labels = [bitstring(i, n_qubits) for i in xs]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    for tick, idx in zip(ax.get_xticklabels(), xs):
        tick.set_color(INK if idx in set(int(i) for i in valid_indices) else "#94a3b8")
    ax.set_xlabel("computational basis  (dark labels = valid BAS)")
    ax.set_ylabel("probability")
    ax.set_title("Target vs trained QCBM")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_model_samples(
    samples: np.ndarray,
    valid_indices: np.ndarray,
    n: int,
    path: Path,
    ncols: int = 8,
) -> Path:
    apply_style()
    n_qubits = n * n
    nrows = int(np.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.35 * ncols + 0.4, 1.45 * nrows + 0.8))
    axes = np.atleast_1d(axes).ravel()
    valid = set(int(i) for i in valid_indices)
    n_valid = 0
    for i, ax in enumerate(axes):
        if i >= len(samples):
            ax.axis("off")
            continue
        idx = int(samples[i])
        bits = np.array(list(bitstring(idx, n_qubits)), dtype=int)
        ok = idx in valid
        n_valid += int(ok)
        _draw_pattern(ax, bits, n, edge=VALID_EDGE if ok else INVALID_EDGE)
    rate = n_valid / max(len(samples), 1)
    fig.suptitle(f"Samples from trained QCBM  ·  {n_valid}/{len(samples)} valid ({rate:.0%})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_circuit_diagram(qnode, weights, path: Path) -> Path:
    apply_style()
    fig, _ax = qml_draw(qnode, weights)
    fig.set_size_inches(11.5, 3.6)
    fig.suptitle("Strongly entangling QCBM ansatz (4 qubits × 4 layers)", color=INK, y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def qml_draw(qnode, weights):
    import pennylane as qml

    try:
        return qml.draw_mpl(qnode, decimals=None, style="black_white", level="device")(weights)
    except TypeError:
        return qml.draw_mpl(qnode, decimals=None, style="black_white")(weights)
