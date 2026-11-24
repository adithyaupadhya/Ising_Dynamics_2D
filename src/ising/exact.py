"""Exact finite-volume Gibbs enumeration for small boxes."""
import numpy as np
from .geometry import SquareLattice

def exact_small(N, beta, lam):
    """Exact Boltzmann law of the N x N box with the - frame (N <= 4)."""
    if not 1 <= N <= 4:
        raise ValueError("Exact enumeration requires 1 <= N <= 4")
    h = lam / N
    geometry = SquareLattice(N)
    M = 1 << geometry.num_sites
    idx = np.arange(M, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(geometry.num_sites)) & 1).astype(np.int8) * 2 - 1
    S = -np.ones((M, *geometry.state_shape), dtype=np.int8)
    geometry.interior(S)[:] = bits.reshape(M, *geometry.interior_shape)
    bonds = sum((a.astype(np.int16) * b).sum(axis=tuple(range(1, geometry.dimension + 1)))
                for a, b in geometry.bond_pairs(S))
    msum = bits.astype(np.int16).sum(axis=1)
    logw = beta * (bonds + h * msum)
    w = np.exp(logw - logw.max())
    return w / w.sum(), bonds, msum
