"""Model parameters and validation."""
from dataclasses import dataclass
import numpy as np

from .geometry.square import SquareLattice, BETA_C as BETA_C

@dataclass(frozen=True)
class IsingModel:
    N: int = 48
    beta: float = .6
    lam: float = 2.

    def __post_init__(self) -> None:
        if not isinstance(self.N, int) or self.N < 1:
            raise ValueError('N must be a positive integer')
        if not np.isfinite(self.beta) or self.beta <= 0 or not np.isfinite(self.lam) or self.lam < 0:
            raise ValueError('Require finite beta > 0 and lambda >= 0')
    @property
    def geometry(self) -> SquareLattice:
        return SquareLattice(self.N)

    @property
    def num_sites(self) -> int:
        return self.geometry.num_sites

    @property
    def num_bonds(self) -> int:
        return self.geometry.num_bonds

    @property
    def h(self) -> float:
        return self.lam / self.N


DYNAMICS = ("heatbath", "metropolis", "sw-ghost", "sw-size")


def validate_algorithm(dynamics: str, backend: str) -> None:
    if dynamics not in DYNAMICS:
        raise ValueError("Unknown dynamics")
    if backend not in ("python", "numba"):
        raise ValueError("Unknown backend")
