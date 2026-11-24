"""Exact finite-volume Gibbs enumeration for small boxes."""
import numpy as np

def exact_small(N, beta, lam):
    """Exact Boltzmann law of the N x N box with the - frame (N <= 4)."""
    if not 1 <= N <= 4:
        raise ValueError("Exact enumeration requires 1 <= N <= 4")
    h = lam / N
    M = 1 << (N * N)
    idx = np.arange(M, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N * N)) & 1).astype(np.int8) * 2 - 1
    S = -np.ones((M, N + 2, N + 2), dtype=np.int8)
    S[:, 1:-1, 1:-1] = bits.reshape(M, N, N)
    bonds = ((S[:, 1:-1, :-1].astype(np.int16) * S[:, 1:-1, 1:]).sum(axis=(1, 2)) +
             (S[:, :-1, 1:-1].astype(np.int16) * S[:, 1:, 1:-1]).sum(axis=(1, 2)))
    msum = bits.astype(np.int16).sum(axis=1)
    logw = beta * (bonds + h * msum)
    w = np.exp(logw - logw.max())
    return w / w.sum(), bonds, msum
