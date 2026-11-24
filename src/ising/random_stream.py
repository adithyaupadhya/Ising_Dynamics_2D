"""Buffered random-site inputs shared by single and coupled chains."""
import numpy as np


class RandomUpdateStream:
    def __init__(self, N: int, rng: np.random.Generator):
        self.N = N
        self.rng = rng
        self._sites = np.empty(0, dtype=np.int64)
        self._uniforms = np.empty(0)

    def peek(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return up to count pending draws without consuming them."""
        if not len(self._sites):
            n = self.N
            ids = self.rng.integers(0, n*n, size=n*n)
            self._sites = (ids//n+1)*(n+2)+ids%n+1
            self._uniforms = self.rng.random(n*n)
        return self._sites[:count], self._uniforms[:count]

    def consume(self, count: int) -> None:
        """Discard exactly the draws used by the transition kernel."""
        if not 0 <= count <= len(self._sites):
            raise ValueError('Invalid consumption count')
        self._sites = self._sites[count:]
        self._uniforms = self._uniforms[count:]
