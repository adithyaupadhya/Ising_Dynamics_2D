"""Square-lattice state initialization (public convenience function)."""
from .geometry import SquareLattice


def make_state(N, init, rng):
    return SquareLattice(N).make_state(init, rng)
