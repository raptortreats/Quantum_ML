"""Bars-and-Stripes (BAS) binary patterns.

An n x n BAS image is valid when every row is uniform (a bar) or every
column is uniform (a stripe). All-zero and all-one belong to both families
and are counted once, so the support size is 2^{n+1} - 2.

2 x 2 BAS has six legal 4-bit strings. That is small enough for a 4-qubit
CPU simulator to return the exact Born distribution (16 amplitudes).
"""

from __future__ import annotations

import numpy as np

N_DEFAULT = 2


def bars_and_stripes(n: int = N_DEFAULT) -> np.ndarray:
    """Return unique n x n BAS patterns as flattened row-major bit vectors."""
    if n < 2:
        raise ValueError("BAS is defined for n >= 2")

    bits = np.array([list(np.binary_repr(i, n))[::-1] for i in range(2**n)], dtype=int)

    stripes = np.repeat(bits, n, axis=0).reshape(2**n, n * n)
    bars = np.repeat(bits.reshape(2**n * n, 1), n, axis=1).reshape(2**n, n * n)

    # Drop the duplicated all-ones stripe and all-zero bar.
    return np.vstack((stripes[:-1], bars[1:]))


def pattern_index(pattern: np.ndarray) -> int:
    """Map a flattened binary pattern to its computational-basis integer."""
    return int("".join(str(int(bit)) for bit in np.asarray(pattern).ravel()), 2)


def bitstring(index: int, n_bits: int) -> str:
    return format(index, f"0{n_bits}b")


def target_distribution(n: int = N_DEFAULT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform distribution over the 2^{n+1}-2 unique BAS patterns.

    Returns
    -------
    patterns
        Integer array of shape (n_valid, n * n).
    valid_indices
        Computational-basis indices of those patterns.
    probs
        Length-2^{n^2} target vector π(x), uniform on the valid support.
    """
    patterns = bars_and_stripes(n)
    n_qubits = n * n
    dim = 2**n_qubits
    valid_indices = np.array([pattern_index(p) for p in patterns], dtype=int)
    probs = np.zeros(dim, dtype=np.float64)
    probs[valid_indices] = 1.0 / len(patterns)
    return patterns, valid_indices, probs
