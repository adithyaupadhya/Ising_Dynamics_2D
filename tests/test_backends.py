import unittest
import numpy as np
from ising import IsingModel, Simulator

class TestBackends(unittest.TestCase):

    def test_identical_trajectories(self):
        for n in (1, 3, 16, 64):
            for dynamics in ('heatbath', 'metropolis'):
                a, b = [Simulator(IsingModel(n, 0.5, 3.0), 'random', 42, dynamics=dynamics, backend=k) for k in ('python', 'numba')]
                for count in (1, n * n, 3, 2 * n * n, 5):
                    a.advance_updates(count)
                    b.advance_updates(count)
                    np.testing.assert_array_equal(a.state, b.state)
                    self.assertEqual(a.magnetization, b.magnetization)
                    self.assertEqual(a.magnetization, a.state[1:-1, 1:-1].mean())

    def test_first_passage_matches_single_steps(self):
        for backend in ('python', 'numba'):
            for dynamics in ('heatbath', 'metropolis', 'sw'):
                for init, direction in [('minus', 1), ('plus', -1)]:
                    c = IsingModel(3, 0.5, 3.0)
                    a, b = (Simulator(c, init, 12, dynamics=dynamics, backend=backend), Simulator(c, init, 12, dynamics=dynamics, backend=backend))
                    hit = a.first_passage(100, 0, direction)
                    limit = 100 if dynamics == 'sw' else 900
                    while b.updates < limit and direction * b.magnetization < 0:
                        b.advance_updates()
                    self.assertEqual(hit, direction * b.magnetization >= 0)
                    self.assertEqual(a.updates, b.updates)
                    np.testing.assert_array_equal(a.state, b.state)
                    a.advance_updates(11)
                    b.advance_updates(11)
                    np.testing.assert_array_equal(a.state, b.state)

    def test_initial_hit_and_censor(self):
        s = Simulator(IsingModel(2), 'plus', 42)
        self.assertTrue(s.first_passage(10, 0, 1))
        self.assertEqual(s.updates, 0)
        self.assertFalse(s.first_passage(0, 0, -1))
        self.assertEqual(s.updates, 0)

    def test_state_replacement(self):
        s = Simulator(IsingModel(2), 'minus', 42)
        with self.assertRaises(ValueError):
            s.state[1, 1] = 1
        state = s.state.copy()
        state[1, 1] = 1
        s.set_state(state)
        self.assertEqual(s.magnetization, -0.5)
        state[0, 0] = 1
        with self.assertRaises(ValueError):
            s.set_state(state)
