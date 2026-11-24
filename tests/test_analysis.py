import unittest
import numpy as np
from ising.analysis import kaplan_meier, median_survival, restricted_mean, exponential_fit, passage_medians, autocorrelation, integrated_time, height_correlation, batch_mean, bootstrap_mean

class TestAnalysis(unittest.TestCase):

    def test_survival_ties_and_censoring(self):
        curve = kaplan_meier([1, 2, 2, 4], [1, 1, 0, 0])
        self.assertEqual([row['at_risk'] for row in curve], [4, 3, 1])
        np.testing.assert_allclose([row['survival'] for row in curve], [0.75, 0.5, 0.5])
        self.assertEqual(median_survival(curve), 2)
        self.assertAlmostEqual(restricted_mean(curve, 4), 2.75)
        self.assertIsNone(median_survival(kaplan_meier([2, 3], [0, 0])))
        self.assertEqual(kaplan_meier([1], [1])[-1]['survival'], 0)

    def test_scaling(self):
        n = np.array([3, 4, 5, 6])
        self.assertAlmostEqual(exponential_fit(n, np.exp(0.4 * n + 1))['rate'], 0.4)
        rows = [dict(N=3, sweeps=2, event=0), dict(N=4, sweeps=3, event=1)]
        self.assertEqual(passage_medians(rows), [dict(N=3, median=None), dict(N=4, median=3)])
        with self.assertRaises(ValueError):
            exponential_fit([1, 2], [1, 2])

    def test_autocorrelation(self):
        x = np.array([1, 3, 2, 5, 1], float)
        y = x - x.mean()
        expected = np.correlate(y, y, 'full')[len(y) - 1:] / (y @ y)
        np.testing.assert_allclose(autocorrelation(x), expected)
        self.assertAlmostEqual(integrated_time([1, 0.2, 0.1, -0.2]), 0.7)
        with self.assertRaises(ValueError):
            autocorrelation([1, 1])

    def test_height_and_uncertainty(self):
        rows = height_correlation([[1, 2, None], [2, 3, 4]], 2)
        self.assertEqual([r['pairs'] for r in rows], [5, 3, 1])
        self.assertAlmostEqual(rows[0]['correlation'], 1)
        self.assertEqual(batch_mean([1, 1, 3, 3, 999], 2), (2.0, 1.0))
        self.assertEqual(bootstrap_mean([2, 2, 2], replicates=20), (2.0, 2.0))
