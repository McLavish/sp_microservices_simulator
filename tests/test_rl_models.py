import unittest

import numpy as np

from rl.models.evt_tail import EVTTailModel
from rl.models.lognormal import LogNormalModel
from rl.models.lognormal_mixture import LogNormalMixtureModel


class TestRLModels(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)

    def test_lognormal_model(self):
        data = self.rng.lognormal(mean=3.0, sigma=0.5, size=1000)
        model = LogNormalModel().fit(data)
        p50 = model.quantile(0.5)
        self.assertGreater(p50, 0.0)
        ll = model.log_likelihood(data)
        self.assertTrue(np.isfinite(ll))

    def test_lognormal_mixture_model(self):
        data = np.concatenate(
            [
                self.rng.lognormal(mean=2.5, sigma=0.4, size=500),
                self.rng.lognormal(mean=4.0, sigma=0.8, size=500),
            ]
        )
        model = LogNormalMixtureModel().fit(data)
        weights = model.weights
        assert weights is not None
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=6)
        p95 = model.quantile(0.95)
        self.assertGreater(p95, 0.0)

    def test_evt_tail_model(self):
        data = self.rng.lognormal(mean=3.0, sigma=0.8, size=1000)
        model = EVTTailModel().fit(data)
        p99 = model.quantile(0.99)
        self.assertGreater(p99, 0.0)
        cdf_val = model.cdf(np.array([p99]))[0]
        self.assertGreaterEqual(cdf_val, 0.9)


if __name__ == "__main__":
    unittest.main()
