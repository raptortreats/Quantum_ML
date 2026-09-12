"""Quantum Circuit Born Machine on a PennyLane CPU simulator.

The model prepares |ψ(θ)⟩ with a hardware-efficient ansatz. Measuring in the
computational basis yields the Born distribution

    p_θ(x) = |⟨x|ψ(θ)⟩|^2.

On 4 qubits the simulator returns the exact 16-outcome probability vector, so
we can train with Kullback–Leibler divergence and a PyTorch Adam loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
import torch

from src.bas import bitstring


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    kls: list[float] = field(default_factory=list)
    validity: list[float] = field(default_factory=list)
    weights: torch.nn.Parameter | None = None


class QCBM:
    """Parameterized Born machine on ``default.qubit``."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 4, seed: int = 7) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        rng = np.random.default_rng(seed)
        init = torch.tensor(rng.uniform(0.0, 2.0 * np.pi, size=shape), dtype=torch.float64)
        self.weights = torch.nn.Parameter(init)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(weights):
            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits),
                ranges=[1] * n_layers,
            )
            return qml.probs(wires=range(n_qubits))

        return circuit

    def probabilities(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        return self.circuit(self.weights if weights is None else weights)

    def kl_divergence(self, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """KL(π || p_θ) with a floor so empty bins stay finite."""
        model = self.probabilities(weights)
        eps = torch.tensor(1e-12, dtype=model.dtype)
        return torch.sum(target * (torch.log(target + eps) - torch.log(model + eps)))


def train_qcbm(
    model: QCBM,
    target_probs: np.ndarray,
    valid_indices: np.ndarray,
    steps: int = 160,
    lr: float = 0.08,
    log_every: int = 20,
) -> TrainResult:
    """Minimize KL(π || p_θ) with Adam on exact simulator probabilities."""
    target = torch.tensor(target_probs, dtype=torch.float64)
    valid = torch.tensor(valid_indices, dtype=torch.long)
    optimizer = torch.optim.Adam([model.weights], lr=lr)
    result = TrainResult(weights=model.weights)

    for step in range(steps):
        optimizer.zero_grad()
        loss = model.kl_divergence(target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = model.probabilities()
            kl = float(loss.detach())
            validity = float(probs[valid].sum())

        result.losses.append(kl)
        result.kls.append(kl)
        result.validity.append(validity)

        if log_every and step % log_every == 0:
            print(f"step {step:3d}  KL={kl:.4f}  P(valid)={validity:.3f}")

    print(f"step {steps - 1:3d}  KL={result.losses[-1]:.4f}  P(valid)={result.validity[-1]:.3f}")
    return result


def sample_bitstrings(probs: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Draw computational-basis integers from a discrete distribution."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(probs), size=n_samples, p=probs)


def evaluate(
    model_probs: np.ndarray,
    target_probs: np.ndarray,
    valid_indices: np.ndarray,
    n_qubits: int,
) -> dict[str, float | list[tuple[str, float, float]]]:
    """Scalar metrics plus per-pattern target vs model probabilities."""
    valid = np.asarray(valid_indices)
    tv = 0.5 * float(np.abs(model_probs - target_probs).sum())
    validity = float(model_probs[valid].sum())
    rows = [
        (bitstring(int(idx), n_qubits), float(target_probs[idx]), float(model_probs[idx]))
        for idx in valid
    ]
    return {
        "kl": float(np.sum(target_probs * (np.log(target_probs + 1e-12) - np.log(model_probs + 1e-12)))),
        "total_variation": tv,
        "validity_mass": validity,
        "valid_pattern_probs": rows,
    }
