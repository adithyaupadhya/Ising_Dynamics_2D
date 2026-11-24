"""Buffered random-site inputs shared by single and coupled chains."""
import numpy as np
from .geometry import SquareLattice


class RandomUpdateStream:
    def __init__(self, N: int | SquareLattice, rng: np.random.Generator):
        self.geometry = SquareLattice(N) if isinstance(N, int) else N
        self.N = self.geometry.N
        self.rng = rng
        self._sites = np.empty(0, dtype=np.int64)
        self._uniforms = np.empty(0)

    def peek(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return up to count pending draws without consuming them."""
        if not len(self._sites):
            num_sites = self.geometry.num_sites
            site_ids = self.rng.integers(0, num_sites, size=num_sites)
            self._sites = self.geometry.padded_indices(site_ids)
            self._uniforms = self.rng.random(num_sites)
        return self._sites[:count], self._uniforms[:count]

    def consume(self, count: int) -> None:
        """Discard exactly the draws used by the transition kernel."""
        if not 0 <= count <= len(self._sites):
            raise ValueError('Invalid consumption count')
        self._sites = self._sites[count:]
        self._uniforms = self._uniforms[count:]
