"""Square-box geometry with a fixed minus exterior and row-major storage."""
from dataclasses import dataclass
import numpy as np

BETA_C = 0.5 * np.log(1 + np.sqrt(2))

# Geometric only: pair N-W and S-E dual edges, joining NE-SW sites.
NW_CONNECTIVITY = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
NW_CONNECTIVITY.setflags(write=False)


def interior(state):
    """Dynamic sites; also supports leading batch axes."""
    return state[..., 1:-1, 1:-1]


def bond_pairs(state):
    """Views of endpoints of every physical edge incident to the interior.

    Horizontal bonds precede vertical bonds; exterior-only edges are omitted.
    Leading batch axes are preserved.
    """
    return ((state[..., 1:-1, :-1], state[..., 1:-1, 1:]),
            (state[..., :-1, 1:-1], state[..., 1:, 1:-1]))


def broken_bonds(state):
    return sum(int(np.count_nonzero(a != b)) for a, b in bond_pairs(state))


@dataclass(frozen=True)
class SquareLattice:
    N: int

    def __post_init__(self):
        if not isinstance(self.N, int) or self.N < 1:
            raise ValueError('N must be a positive integer')

    @property
    def dimension(self):
        return 2

    @property
    def degree(self):
        return 4

    @property
    def num_sites(self):
        return self.N**2

    @property
    def num_bonds(self):
        return 2 * self.N * (self.N + 1)

    @property
    def state_shape(self):
        return (self.N + 2, self.N + 2)

    @property
    def interior_shape(self):
        return (self.N, self.N)

    @property
    def beta_c(self):
        return BETA_C

    interior = staticmethod(interior)
    bond_pairs = staticmethod(bond_pairs)
    broken_bonds = staticmethod(broken_bonds)

    def padded_indices(self, site_ids):
        """Map contiguous dynamic vertex IDs to flattened padded indices."""
        return (site_ids // self.N + 1) * (self.N + 2) + site_ids % self.N + 1

    def make_state(self, init, rng):
        spins = -np.ones(self.state_shape, dtype=np.int8)
        if init == 'plus':
            self.interior(spins)[:] = 1
        elif init == 'random':
            self.interior(spins)[:] = rng.choice([-1, 1], size=self.interior_shape).astype(np.int8)
        elif init != 'minus':
            raise ValueError(init)
        return spins

    def validate_state(self, state):
        if (state.shape != self.state_shape or not np.isin(state, [-1, 1]).all()
                or not np.all(state[[0, -1], :] == -1)
                or not np.all(state[:, [0, -1]] == -1)):
            raise ValueError('Invalid padded spin configuration')

    @property
    def local_kernel_args(self):
        """Geometry-specific arguments to the existing compiled local kernels."""
        return (self.N + 2,)

    def local_kernel(self, backend):
        from ..kernels.local import random_updates_python, random_updates_numba
        return random_updates_numba if backend == 'numba' else random_updates_python

    def coupled_kernel(self, backend):
        from ..kernels.local import coupled_updates_python, coupled_updates_numba
        return coupled_updates_numba if backend == 'numba' else coupled_updates_python

    def cluster_kernel(self, dynamics, backend):
        from ..kernels.cluster import CLUSTER_KERNELS
        return CLUSTER_KERNELS.get((dynamics, backend))

    @property
    def observables(self):
        """Current square-specific observable implementation (including NW splitting)."""
        from .. import observables
        return observables
