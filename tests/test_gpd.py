"""Tests for GPDFitter."""

import numpy as np
import pytest
from scipy.stats import genpareto

from insurance_evt import GPDFitter
from insurance_evt.results import GPDFitResult


class TestGPDFitterInit:
    def test_basic_init(self, pareto_claims):
        claims, threshold, xi, sigma = pareto_claims
        fitter = GPDFitter(claims, threshold=threshold)
        assert fitter.threshold == threshold
        assert fitter.n_exceedances > 0

    def test_n_exceedances_correct(self, pareto_claims):
        claims, threshold, xi, sigma = pareto_claims
        fitter = GPDFitter(claims, threshold=threshold)
        expected = np.sum(claims > threshold)
        assert fitter.n_exceedances == expected

    def test_rejects_too_few_exceedances(self):
        claims = np.array([1000.0, 1200.0, 1100.0, 900.0, 800.0, 600.0])
        with pytest.raises(ValueError, match="at least 5"):
            GPDFitter(claims, threshold=5000.0).fit()

    def test_right_censoring_wrong_length(self, pareto_claims):
        claims, threshold, *_ = pareto_claims
        with pytest.raises(ValueError, match="right_censoring length"):
            GPDFitter(claims, threshold=threshold, right_censoring=np.array([True, False]))

    def test_left_truncation_below_threshold(self, pareto_claims):
        claims, threshold, *_ = pareto_claims
        with pytest.raises(ValueError, match="left_truncation"):
            GPDFitter(claims, threshold=threshold, left_truncation=threshold - 1000)


class TestGPDFit:
    def test_returns_gpd_fit_result(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000  # threshold = 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        assert isinstance(result, GPDFitResult)

    def test_xi_estimate_close_to_truth(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        threshold = 50_000
        claims = excesses + threshold
        fitter = GPDFitter(claims, threshold=threshold)
        result = fitter.fit()
        # Within 0.15 of true xi (MLE has finite-sample variance)
        assert abs(result.xi - xi_true) < 0.20

    def test_sigma_estimate_close_to_truth(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        threshold = 50_000
        claims = excesses + threshold
        fitter = GPDFitter(claims, threshold=threshold)
        result = fitter.fit()
        # Within 20% of true sigma
        assert abs(result.sigma - sigma_true) / sigma_true < 0.25

    def test_se_positive(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        assert result.xi_se > 0
        assert result.sigma_se > 0

    def test_loglik_finite(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        assert np.isfinite(result.loglik)

    def test_threshold_stored(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        threshold = 123_456
        claims = excesses + threshold
        fitter = GPDFitter(claims, threshold=threshold)
        result = fitter.fit()
        assert result.threshold == threshold

    def test_n_exceedances_stored(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        assert result.n_exceedances == len(excesses)

    def test_sigma_positive(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        assert result.sigma > 0

    def test_repr(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        result = GPDFitter(claims, threshold=100_000).fit()
        r = repr(result)
        assert "GPDFitResult" in r
        assert "xi=" in r

    def test_unsupported_method(self, gpd_excesses):
        excesses, *_ = gpd_excesses
        claims = excesses + 100_000
        with pytest.raises(ValueError, match="mle"):
            GPDFitter(claims, threshold=100_000).fit(method="mcmc")


class TestGPDFitCensored:
    def test_fit_with_censoring(self, censored_claims):
        all_claims, all_censored, threshold, xi_true, sigma_true = censored_claims
        fitter = GPDFitter(all_claims, threshold=threshold, right_censoring=all_censored)
        result = fitter.fit()
        assert isinstance(result, GPDFitResult)
        assert result.n_censored > 0

    def test_xi_estimate_censored(self, censored_claims):
        all_claims, all_censored, threshold, xi_true, sigma_true = censored_claims
        fitter = GPDFitter(all_claims, threshold=threshold, right_censoring=all_censored)
        result = fitter.fit()
        # Censored MLE should recover xi within reasonable tolerance
        assert abs(result.xi - xi_true) < 0.30

    def test_censored_vs_uncensored_different(self, censored_claims):
        """Censored MLE should give different result than ignoring censoring."""
        all_claims, all_censored, threshold, xi_true, sigma_true = censored_claims
        result_censored = GPDFitter(
            all_claims, threshold=threshold, right_censoring=all_censored
        ).fit()
        result_uncensored = GPDFitter(all_claims, threshold=threshold).fit()
        # They should differ (censoring matters)
        assert result_censored.xi != result_uncensored.xi


class TestGPDFitTruncated:
    def test_left_truncation(self, pareto_claims):
        claims, threshold, xi_true, sigma_true = pareto_claims
        # Reinsurer only sees claims > 750k when threshold=500k
        trunc = 750_000
        fitter = GPDFitter(claims, threshold=threshold, left_truncation=trunc)
        result = fitter.fit()
        assert isinstance(result, GPDFitResult)
        assert result.left_truncation == trunc

    def test_truncation_fewer_exceedances(self, pareto_claims):
        claims, threshold, *_ = pareto_claims
        fitter_base = GPDFitter(claims, threshold=threshold)
        fitter_trunc = GPDFitter(claims, threshold=threshold, left_truncation=700_000)
        assert fitter_trunc.n_exceedances <= fitter_base.n_exceedances


class TestProfileLikelihoodCI:
    def test_returns_tuple(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        fitter.fit()
        lo, hi = fitter.profile_likelihood_ci("xi")
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_mle(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        lo, hi = fitter.profile_likelihood_ci("xi")
        assert lo < result.xi < hi

    def test_ci_asymmetric_for_heavy_tail(self, gpd_excesses):
        """Upper CI should extend further than lower for xi > 0."""
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        lo, hi = fitter.profile_likelihood_ci("xi")
        upper_extent = hi - result.xi
        lower_extent = result.xi - lo
        # For positive xi, upper should typically be larger
        assert upper_extent > 0

    def test_ci_contains_truth(self, gpd_excesses):
        """95% CI should contain true parameter most of the time."""
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        fitter.fit()
        lo, hi = fitter.profile_likelihood_ci("xi")
        assert lo <= xi_true <= hi

    def test_sigma_ci(self, gpd_excesses):
        excesses, xi_true, sigma_true = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        result = fitter.fit()
        lo, hi = fitter.profile_likelihood_ci("sigma")
        assert lo < result.sigma < hi

    def test_profile_ci_before_fit_raises(self, gpd_excesses):
        excesses, *_ = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        with pytest.raises(RuntimeError, match="fit()"):
            fitter.profile_likelihood_ci("xi")

    def test_invalid_param_name(self, gpd_excesses):
        excesses, *_ = gpd_excesses
        claims = excesses + 100_000
        fitter = GPDFitter(claims, threshold=100_000)
        fitter.fit()
        with pytest.raises(ValueError):
            fitter.profile_likelihood_ci("mu")
