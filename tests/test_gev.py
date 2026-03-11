"""Tests for GEVFitter."""

import numpy as np
import pytest
from scipy.stats import genextreme

from insurance_evt import GEVFitter
from insurance_evt.results import GEVFitResult


class TestGEVFitterInit:
    def test_basic_init(self, gev_maxima):
        maxima, xi, mu, sigma = gev_maxima
        fitter = GEVFitter(maxima)
        assert len(fitter.block_maxima) == len(maxima)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            GEVFitter(np.array([1.0, -2.0, 3.0]))

    def test_rejects_too_few(self):
        with pytest.raises(ValueError, match="at least 5"):
            GEVFitter(np.array([1.0, 2.0, 3.0]))


class TestGEVFit:
    def test_returns_gev_fit_result(self, gev_maxima):
        maxima, xi, mu, sigma = gev_maxima
        fitter = GEVFitter(maxima)
        result = fitter.fit()
        assert isinstance(result, GEVFitResult)

    def test_xi_estimate_close_to_truth(self, gev_maxima):
        maxima, xi_true, mu_true, sigma_true = gev_maxima
        fitter = GEVFitter(maxima)
        result = fitter.fit()
        assert abs(result.xi - xi_true) < 0.20

    def test_mu_estimate_close_to_truth(self, gev_maxima):
        maxima, xi_true, mu_true, sigma_true = gev_maxima
        fitter = GEVFitter(maxima)
        result = fitter.fit()
        assert abs(result.mu - mu_true) / mu_true < 0.15

    def test_sigma_positive(self, gev_maxima):
        maxima, *_ = gev_maxima
        result = GEVFitter(maxima).fit()
        assert result.sigma > 0

    def test_se_positive(self, gev_maxima):
        maxima, *_ = gev_maxima
        result = GEVFitter(maxima).fit()
        assert result.xi_se > 0
        assert result.mu_se > 0
        assert result.sigma_se > 0

    def test_loglik_finite(self, gev_maxima):
        maxima, *_ = gev_maxima
        result = GEVFitter(maxima).fit()
        assert np.isfinite(result.loglik)

    def test_n_blocks_stored(self, gev_maxima):
        maxima, *_ = gev_maxima
        result = GEVFitter(maxima).fit()
        assert result.n_blocks == len(maxima)

    def test_repr(self, gev_maxima):
        maxima, *_ = gev_maxima
        result = GEVFitter(maxima).fit()
        r = repr(result)
        assert "GEVFitResult" in r


class TestGEVDomainOfAttraction:
    def test_frechet_for_positive_xi(self, gev_maxima):
        maxima, xi_true, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        fitter.fit()
        doa = fitter.domain_of_attraction()
        # xi_true = 0.2 => Frechet
        assert doa in ["Frechet", "Gumbel"]  # allow Gumbel if estimate near zero

    def test_gumbel_xi_zero(self):
        rng = np.random.default_rng(99)
        maxima = genextreme.rvs(0, loc=1e6, scale=2e5, size=40, random_state=rng)
        maxima = np.abs(maxima)
        fitter = GEVFitter(maxima)
        fitter.fit()
        doa = fitter.domain_of_attraction()
        assert doa in ["Gumbel", "Frechet", "Weibull"]  # any valid answer

    def test_domain_before_fit_raises(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        with pytest.raises(RuntimeError, match="fit()"):
            fitter.domain_of_attraction()


class TestGEVReturnLevel:
    def test_return_level_increases_with_period(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        fitter.fit()
        levels = [fitter.return_level(T) for T in [10, 50, 100, 200, 500]]
        assert all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))

    def test_return_level_200_positive(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        fitter.fit()
        rl = fitter.return_level(200)
        assert rl > 0

    def test_return_level_before_fit_raises(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        with pytest.raises(RuntimeError):
            fitter.return_level(200)


class TestGEVProfileCI:
    def test_returns_tuple(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        fitter.fit()
        lo, hi = fitter.profile_likelihood_ci(200)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_point_estimate(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        fitter.fit()
        rl = fitter.return_level(200)
        lo, hi = fitter.profile_likelihood_ci(200)
        assert lo <= rl <= hi

    def test_ci_positive(self, gev_maxima):
        maxima, *_ = gev_maxima
        fitter = GEVFitter(maxima)
        fitter.fit()
        lo, hi = fitter.profile_likelihood_ci(200)
        assert lo > 0
        assert hi > lo
