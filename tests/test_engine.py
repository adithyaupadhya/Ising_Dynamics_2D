import unittest
import numpy as np
from ising import IsingModel, Simulator
from ising.exact import exact_small
from ising.experiments.trajectory import run

class TestEngine(unittest.TestCase):

    def test_boundaries_and_reproducibility(self):
        for n in (1, 3, 4):
            for d in ('heatbath', 'metropolis', 'sw'):
                for backend in ('python', 'numba'):
                    a = Simulator(IsingModel(n, 0.5, 3.0), 'random', 4, dynamics=d, backend=backend)
                    b = Simulator(a.model, 'random', 4, dynamics=d, backend=backend)
                    a.advance(20)
                    b.advance(20)
                    np.testing.assert_array_equal(a.state, b.state)
                    self.assertTrue(np.all(a.state[[0, -1], :] == -1))
                    self.assertTrue(np.all(a.state[:, [0, -1]] == -1))
                    self.assertTrue(np.isin(a.state, [-1, 1]).all())
                    self.assertEqual(a.t, 20)

    def test_clusters(self):
        a = Simulator(IsingModel(3), 'minus', 0)
        self.assertEqual(a.observe()['largest_plus_fraction'], 0)
        state = a.state.copy()
        state[1, 1:3] = 1
        state[3, 3] = 1
        a.set_state(state)
        self.assertAlmostEqual(a.observe()['largest_plus_fraction'], 2 / 9)

    def test_time(self):
        _, s = run(IsingModel(3, 0.5, 3.0), 7, record_every=3)
        self.assertEqual([x['sweep'] for x in s], [0, 3, 6, 7])
        self.assertEqual(s[0]['magnetization'], -1)

    def test_exact_one_site(self):
        beta = 0.5
        lam = 3.0
        w, _, _ = exact_small(1, beta, lam)
        expected = 1 / (1 + np.exp(-2 * beta * (lam - 4)))
        self.assertAlmostEqual(w[1], expected)
        for d in ('heatbath', 'metropolis', 'sw'):
            a = Simulator(IsingModel(1, beta, lam), 'minus', 52, dynamics=d)
            a.advance(100)
            vals = []
            for _ in range(10000):
                a.advance()
                vals.append(a.magnetization > 0)
            self.assertLess(abs(np.mean(vals) - expected), 0.045, d)

    def test_exact_bond_convention(self):
        from ising.observables import bond_sum
        for n in (1, 2, 3):
            w, bonds, _ = exact_small(n, 0.5, 3.0)
            self.assertAlmostEqual(w.sum(), 1.0)
            self.assertEqual(bonds[0], 2 * n * (n + 1))
            sim = Simulator(IsingModel(n), 'plus', 0)
            self.assertEqual(bonds[-1], bond_sum(sim.state))
            self.assertEqual(bonds[-1], 2 * n * (n + 1) - 8 * n)

    def test_validation(self):
        for kw in ({'N': 0}, {'lam': -1}, {'beta': float('nan')}):
            with self.assertRaises(ValueError):
                IsingModel(**kw)
        with self.assertRaises(ValueError):
            Simulator(dynamics='oops')
        with self.assertRaises(ValueError):
            exact_small(5, 0.5, 3.0)
if __name__ == '__main__':
    unittest.main()
