"""Check field-dependent cluster transitions against one-site exact formulas."""
import unittest
import numpy as np
from ising.kernels import CLUSTER_KERNELS


class TestClusterKernels(unittest.TestCase):
    def test_one_site_transition_probabilities(self):
        beta, h = .25, 1.5
        free = np.exp(-8 * beta)
        logistic = 1 / (1 + np.exp(-2 * beta * h))
        for (dynamics, backend), kernel in CLUSTER_KERNELS.items():
            for initial in (-1, 1):
                if dynamics == 'sw-size':
                    expected = logistic * (free if initial == -1 else 1)
                else:
                    expected = free / 2 if initial == -1 else 1 - np.exp(-2 * beta * h) / 2
                with self.subTest(dynamics=dynamics, backend=backend, initial=initial):
                    rng = np.random.default_rng(123)
                    state = np.full((3, 3), -1, dtype=np.int8)
                    count = 0
                    for _ in range(5000):
                        state[1, 1] = initial
                        kernel(state, beta, h, rng)
                        count += state[1, 1] == 1
                    self.assertLess(abs(count / 5000 - expected), .025)
                    self.assertTrue(np.all(state[[0, -1], :] == -1))
                    self.assertTrue(np.all(state[:, [0, -1]] == -1))
