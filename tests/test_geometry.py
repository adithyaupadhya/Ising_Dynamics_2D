"""Square geometry contract and its use by the shared engine."""
import numpy as np
import pytest
from ising import IsingModel, Simulator, MonotoneCoupling
from ising.geometry import SquareLattice
from ising.geometry.square import NW_CONNECTIVITY
from ising.observables import NW_CONNECTIVITY as PUBLIC_CONNECTIVITY
from ising.observables import bond_sum, interface_length, thermodynamics
from ising.random_stream import RandomUpdateStream
from ising.state import make_state


@pytest.mark.parametrize('N', [1, 2, 5])
def test_square_geometry(N):
    geometry = SquareLattice(N)
    model = IsingModel(N)
    assert model.geometry == geometry
    assert model.num_sites == N*N
    assert model.num_bonds == 2*N*(N+1)
    assert geometry.degree == 4
    assert geometry.dimension == 2
    assert PUBLIC_CONNECTIVITY is NW_CONNECTIVITY
    rng = np.random.default_rng(42)
    state = geometry.make_state('random', rng)
    geometry.validate_state(state)
    assert state.shape == geometry.state_shape
    assert np.shares_memory(geometry.interior(state), state)
    np.testing.assert_array_equal(
        geometry.padded_indices(np.arange(geometry.num_sites)),
        np.arange(state.size).reshape(state.shape)[1:-1, 1:-1].ravel())
    assert sum(a.size for a, _ in geometry.bond_pairs(state)) == geometry.num_bonds
    assert bond_sum(state) == geometry.num_bonds - 2*geometry.broken_bonds(state)
    assert interface_length(state) == geometry.broken_bonds(state)
    plus = geometry.make_state('plus', rng)
    assert geometry.broken_bonds(plus) == 4*N
    plus[0, 0] = 1
    with pytest.raises(ValueError):
        geometry.validate_state(plus)


def test_stream_preserves_original_mapping_and_draw_order():
    N = 5
    seed = 73
    reference = np.random.default_rng(seed)
    ids = reference.integers(0, N*N, size=N*N)
    expected_sites = (ids//N+1)*(N+2)+ids%N+1
    expected_uniforms = reference.random(N*N)
    for size in (N, SquareLattice(N)):
        stream = RandomUpdateStream(size, np.random.default_rng(seed))
        sites, uniforms = stream.peek(7)
        np.testing.assert_array_equal(sites, expected_sites[:7])
        np.testing.assert_array_equal(uniforms, expected_uniforms[:7])
        stream.consume(7)
        sites, uniforms = stream.peek(N*N)
        np.testing.assert_array_equal(sites, expected_sites[7:])
        np.testing.assert_array_equal(uniforms, expected_uniforms[7:])
    np.testing.assert_array_equal(make_state(N, 'random', np.random.default_rng(seed)),
                                  SquareLattice(N).make_state('random', np.random.default_rng(seed)))


def test_engine_geometry_bookkeeping():
    model = IsingModel(4, .5, 3)
    sim = Simulator(model, 'random', 3, backend='python')
    for count in (1, 3, 16):
        sim.advance_updates(count)
        state = sim.state
        expected = thermodynamics(model.N, model.h, int(state[1:-1, 1:-1].sum()),
                                   interface_length(state))
        actual = sim.observe(clusters=False)
        assert all(actual[key] == val for key, val in expected.items())
        assert sim.t == sim.updates / model.num_sites
    coupling = MonotoneCoupling(model, seed=3, backend='python')
    coupling.advance_updates(11)
    assert coupling.t == coupling.updates / model.num_sites
    assert coupling.disagreement_fraction == np.mean(coupling.lower[1:-1, 1:-1] != coupling.upper[1:-1, 1:-1])
