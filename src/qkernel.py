"""Aer-backed quantum kernel + QSVC helpers (Qiskit Machine Learning 0.9).

``QuantumKernel`` was removed in Qiskit ML 0.8+. The current objects are
``FidelityQuantumKernel`` / ``QSVC``. For a 4-qubit CPU demo we evaluate the
same fidelity kernel

    K(x, y) = |⟨φ(x) | φ(y)⟩|²

exactly with ``AerSimulator(method="statevector")`` and pass that kernel into
``QSVC``. That is the infinite-shot limit of compute-uncompute on Aer — no
hardware backend, no IBM Runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import BaseKernel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC


def make_zz_feature_map(n_features: int = 4, reps: int = 2, entanglement: str = "linear"):
    """Current Qiskit 2.x factory (``zz_feature_map``); 4 qubits for 4 features."""
    return zz_feature_map(
        feature_dimension=n_features,
        reps=reps,
        entanglement=entanglement,
        name="ZZFeatureMap",
    )


class AerFidelityKernel(BaseKernel):
    """``BaseKernel`` that computes exact statevector fidelities on Aer.

    Statevectors are cached per feature tuple so train/test evaluations reuse
    the same Aer shots (here: exact amplitudes, no sampling noise).
    """

    def __init__(self, feature_map, *, seed: int = 21, enforce_psd: bool = True) -> None:
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)
        self._sim = AerSimulator(method="statevector", seed_simulator=seed)
        self._param_order = sorted(feature_map.parameters, key=lambda p: p.name)
        self._cache: dict[tuple[float, ...], np.ndarray] = {}
        self.backend_name = self._sim.name

    def clear_cache(self) -> None:
        self._cache.clear()

    def _statevector(self, x: np.ndarray) -> np.ndarray:
        key = tuple(float(v) for v in np.asarray(x, dtype=float).ravel())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        bind = {param: value for param, value in zip(self._param_order, key)}
        circuit = self.feature_map.assign_parameters(bind).copy()
        circuit.save_statevector()
        result = self._sim.run(circuit).result()
        vector = np.asarray(result.get_statevector(), dtype=np.complex128)
        self._cache[key] = vector
        return vector

    def _stack(self, data: np.ndarray) -> np.ndarray:
        return np.vstack([self._statevector(row) for row in data])

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x_vec, y_vec = self._validate_input(x_vec, y_vec)
        left = self._stack(x_vec)
        right = left if y_vec is None else self._stack(y_vec)
        kernel = np.abs(left.conj() @ right.T) ** 2
        if y_vec is None and self.enforce_psd:
            kernel = self._make_psd(0.5 * (kernel + kernel.T))
            np.fill_diagonal(kernel, 1.0)
        return np.asarray(kernel.real, dtype=np.float64)


def build_qsvc(kernel: BaseKernel, *, C: float = 1.0) -> QSVC:
    """Qiskit ML QSVC — sklearn SVC with a quantum kernel."""
    return QSVC(quantum_kernel=kernel, C=C)


@dataclass
class KernelFit:
    kernel: AerFidelityKernel
    qsvc: QSVC
    k_train: np.ndarray
    k_test: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    metrics: dict = field(default_factory=dict)
    classical_metrics: dict = field(default_factory=dict)
    classical_pred: np.ndarray | None = None
    classical_score: np.ndarray | None = None


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
    }


def fit_quantum_kernel_qsvc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    reps: int = 2,
    entanglement: str = "linear",
    C: float = 1.0,
    seed: int = 21,
) -> KernelFit:
    feature_map = make_zz_feature_map(n_features=X_train.shape[1], reps=reps, entanglement=entanglement)
    kernel = AerFidelityKernel(feature_map, seed=seed)
    qsvc = build_qsvc(kernel, C=C)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    y_score = qsvc.decision_function(X_test)
    k_train = kernel.evaluate(X_train)
    k_test = kernel.evaluate(X_test, X_train)

    classical = SVC(kernel="rbf", C=C, gamma="scale")
    classical.fit(X_train, y_train)
    c_pred = classical.predict(X_test)
    c_score = classical.decision_function(X_test)

    return KernelFit(
        kernel=kernel,
        qsvc=qsvc,
        k_train=k_train,
        k_test=k_test,
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score, dtype=np.float64),
        metrics=classification_metrics(y_test, y_pred, y_score),
        classical_metrics=classification_metrics(y_test, c_pred, c_score),
        classical_pred=np.asarray(c_pred),
        classical_score=np.asarray(c_score, dtype=np.float64),
    )


def nearest_train_neighbors(k_test: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each test row, index of the most similar training molecule and its label."""
    neighbor_idx = np.argmax(k_test, axis=1)
    neighbor_label = y_train[neighbor_idx]
    return neighbor_idx, neighbor_label
