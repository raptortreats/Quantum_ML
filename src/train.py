"""Train the 2x2 QCBM and write README / notebook figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.bas import target_distribution
from src.plots import (
    save_circuit_diagram,
    save_loss_curve,
    save_model_samples,
    save_probability_comparison,
    save_target_patterns,
)
from src.qcbm import QCBM, evaluate, sample_bitstrings, train_qcbm

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

N = 2
N_LAYERS = 4
STEPS = 160
LR = 0.08
SEED = 7
N_SAMPLES = 32


def run(
    n: int = N,
    n_layers: int = N_LAYERS,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = SEED,
    figures_dir: Path = FIGURES,
) -> dict:
    torch.manual_seed(seed)
    patterns, valid_indices, target_probs = target_distribution(n)
    n_qubits = n * n
    model = QCBM(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_target_patterns(patterns, n, figures_dir / "target_bas_patterns.png")
    save_circuit_diagram(model.circuit, model.weights.detach(), figures_dir / "circuit.png")

    history = train_qcbm(model, target_probs, valid_indices, steps=steps, lr=lr)
    model_probs = model.probabilities().detach().cpu().numpy()
    metrics = evaluate(model_probs, target_probs, valid_indices, n_qubits)
    samples = sample_bitstrings(model_probs, N_SAMPLES, seed=seed + 1)

    save_loss_curve(history.losses, history.validity, figures_dir / "training_loss.png")
    save_probability_comparison(
        target_probs,
        model_probs,
        valid_indices,
        n_qubits,
        figures_dir / "probability_comparison.png",
    )
    save_model_samples(samples, valid_indices, n, figures_dir / "model_samples.png")

    print("metrics:", {k: metrics[k] for k in ("kl", "total_variation", "validity_mass")})
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "patterns": patterns,
        "valid_indices": valid_indices,
        "target_probs": target_probs,
        "model_probs": model_probs,
        "samples": samples,
    }


if __name__ == "__main__":
    run()
