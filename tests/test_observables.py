import unittest
import numpy as np
from ising import IsingModel, Simulator, MonotoneCoupling
from ising.observables import interface_length, bond_sum

class TestObservables(unittest.TestCase):

    def test_incremental_energy(self):
        for dyn in ('heatbath', 'metropolis', 'sw-ghost', 'sw-size'):
            for backend in ('python', 'numba'):
                s = Simulator(IsingModel(5, 0.5, 3), 'random', 42, dynamics=dyn, backend=backend)
                for _ in range(35):
                    s.advance_updates(3)
                    o = s.observe(clusters=False)
                    self.assertEqual(o['interface_length'], interface_length(s.state))
                    self.assertAlmostEqual(o['energy'], -bond_sum(s.state) - s.model.h * s.magnetization * 25)
                s.set_state(-np.ones_like(s.state))
                self.assertEqual(s.observe()['excess_energy'], 0)

    def test_ring_and_profiles(self):
        s = Simulator(IsingModel(7), 'minus', 1)
        state = s.state.copy()
        state[2:5, 2:5] = 1
        state[3, 3] = -1
        state[6, 6] = 1
        s.set_state(state)
        o = s.observe(cluster_details=True, contours=True, boundary_profile=True, interface_heights=True, min_cluster_size=2)
        self.assertEqual(o['cluster_sizes'], [8, 1])
        self.assertEqual(o['clusters_above_threshold'], 1)
        self.assertEqual(o['droplet_area'], 9)
        self.assertEqual(o['droplet_perimeter'], 12)
        self.assertEqual(o['droplet_center'], [3, 3])
        self.assertEqual(o['interface_heights']['top']['heights'], [None, 2, 2, 2, None, None, None])
        self.assertEqual(o['interface_heights']['bottom']['mean'], 4)
        self.assertEqual(o['boundary_magnetization'][0], -1)

    def test_empty_and_full(self):
        for n in (1, 4):
            for init in ('minus', 'plus'):
                s = Simulator(IsingModel(n), init, 0)
                o = s.observe(cluster_details=True, contours=True, interface_heights=True)
                self.assertEqual(o['droplet_area'], n * n if init == 'plus' else 0)
                self.assertEqual(o['droplet_perimeter'], 4 * n if init == 'plus' else 0)
                self.assertEqual(o['interface_heights']['left']['mean'], 1 if init == 'plus' else None)

    def test_coupling(self):
        a, b = [MonotoneCoupling(IsingModel(3, 0.5, 3), 42, backend=backend) for backend in ('python', 'numba')]
        lower = Simulator(a.model, 'minus', 42)
        upper = Simulator(a.model, 'plus', 42)
        for _ in range(5000):
            a.advance_updates()
            b.advance_updates()
            lower.advance_updates()
            upper.advance_updates()
            np.testing.assert_array_equal(a.lower, b.lower)
            np.testing.assert_array_equal(a.upper, b.upper)
            np.testing.assert_array_equal(a.lower, lower.state)
            np.testing.assert_array_equal(a.upper, upper.state)
            self.assertTrue(np.all(a.lower <= a.upper))
            self.assertEqual(a.disagreement_fraction, np.count_nonzero(a.lower != a.upper) / 9)
            if a.coalesced:
                break
        self.assertTrue(a.coalesced)
        t = a.updates
        a.advance(100)
        self.assertEqual(a.updates, t)
        c = MonotoneCoupling(a.model, 42).advance(1000)
        self.assertEqual(c.updates, t)
        with self.assertRaises(ValueError):
            MonotoneCoupling(backend='invalid')
