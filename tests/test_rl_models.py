import math
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

    def test_lognormal_censored_fit_recovers_parameters(self):
        true_mu, true_sigma = 3.0, 0.5
        x = self.rng.lognormal(mean=true_mu, sigma=true_sigma, size=1500)
        cutoff = float(np.quantile(x, 0.6))  # right-censor the slowest 40%
        censored = x > cutoff
        observed = np.minimum(x, cutoff)

        censored_fit = LogNormalModel().fit_censored(observed, censored)
        naive_fit = LogNormalModel().fit(observed[~censored])

        # Censored MLE recovers the true parameters; dropping censored samples
        # is badly biased optimistic (naive mu ~2.7 for these settings).
        self.assertLess(abs(censored_fit.mu - true_mu), 0.1)
        self.assertLess(abs(censored_fit.sigma - true_sigma), 0.1)
        self.assertLess(abs(censored_fit.mu - true_mu), abs(naive_fit.mu - true_mu))

    def test_lognormal_censored_fit_without_censored_samples(self):
        x = self.rng.lognormal(mean=3.0, sigma=0.5, size=500)
        flags = np.zeros(x.size, dtype=bool)
        model = LogNormalModel().fit_censored(x, flags)
        reference = LogNormalModel().fit(x)
        self.assertAlmostEqual(model.mu, reference.mu, places=12)
        self.assertAlmostEqual(model.sigma, reference.sigma, places=12)

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

    def test_evt_body_density_is_evaluated_in_log_space(self):
        """A tight body fit puts a low outlier tens of sigmas out: going
        through the density (exp then log) underflows to 0.0 and reports -inf,
        so one such sample wipes out the whole log-likelihood. The tail branch
        is already computed in log space; the body must be too."""
        x = self.rng.lognormal(mean=np.log(60.0), sigma=0.012, size=5000)
        x[0] = 20.0
        model = EVTTailModel().fit(x)

        body_logpdf = float(model.body.logpdf(np.array([20.0]))[0])
        self.assertLess(body_logpdf, -745.0)  # exp() of this is 0.0 in float64
        self.assertTrue(np.isfinite(body_logpdf))

        logpdf = float(model.logpdf(np.array([20.0]))[0])
        self.assertTrue(np.isfinite(logpdf))
        self.assertTrue(np.isfinite(model.log_likelihood(x)))

        # Exactly the value the probability-space form intends, minus the
        # underflow: body density renormalized onto the body's mass share.
        expected = body_logpdf + math.log(model.p_u / model._body_cdf_u)
        self.assertAlmostEqual(logpdf, expected, places=9)


if __name__ == "__main__":
    unittest.main()
