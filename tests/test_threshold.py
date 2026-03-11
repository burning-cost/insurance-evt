"""Tests for ThresholdSelector."""

import numpy as np
import pytest
from scipy.stats import genpareto

from insurance_evt import ThresholdSelector
from insurance_evt.results import MRLResult, StabilityResult, HillResult


class TestThresholdSelectorInit:
    def test_basic_init(self, pareto_claims):
        claims, threshold, xi, sigma = pareto_claims
        sel = ThresholdSelector(claims)
        assert len(sel.claims) == len(claims)

    def test_rejects_non_positive(self):
        claims = np.array([100.0, -50.0, 200.0])
        with pytest.raises(ValueError, match="strictly positive"):
            ThresholdSelector(claims)

    def test_sorted_descending(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        assert sel._sorted[0] >= sel._sorted[-1]


class TestMRLPlot:
    def test_returns_mrl_result(self, pareto_claims):
        claims, threshold, xi, sigma = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.mrl_plot()
        assert isinstance(result, MRLResult)

    def test_shape(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.mrl_plot(thresholds=np.linspace(100_000, 800_000, 20))
        assert len(result.thresholds) == 20
        assert len(result.mean_excess) == 20
        assert len(result.lower_ci) == 20
        assert len(result.n_above) == 20

    def test_mean_excess_positive(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.mrl_plot()
        valid = ~np.isnan(result.mean_excess)
        assert np.all(result.mean_excess[valid] > 0)

    def test_ci_ordering(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.mrl_plot()
        valid = ~np.isnan(result.mean_excess)
        assert np.all(result.lower_ci[valid] <= result.mean_excess[valid])
        assert np.all(result.mean_excess[valid] <= result.upper_ci[valid])

    def test_to_dataframe(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.mrl_plot()
        df = result.to_dataframe()
        assert "threshold" in df.columns
        assert "mean_excess" in df.columns
        assert "n_above" in df.columns

    def test_n_above_decreasing(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        thresholds = np.linspace(100_000, 900_000, 15)
        result = sel.mrl_plot(thresholds=thresholds)
        # n_above should be non-increasing as threshold increases
        assert np.all(np.diff(result.n_above) <= 0)


class TestParameterStabilityPlot:
    def test_returns_stability_result(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.parameter_stability_plot()
        assert isinstance(result, StabilityResult)

    def test_xi_in_plausible_range(self, pareto_claims):
        claims, threshold, xi_true, sigma = pareto_claims
        sel = ThresholdSelector(claims)
        thresholds = np.linspace(300_000, 700_000, 12)
        result = sel.parameter_stability_plot(thresholds=thresholds)
        valid = ~np.isnan(result.xi_values)
        if np.any(valid):
            median_xi = np.nanmedian(result.xi_values)
            # Should be in reasonable range of true xi=0.3
            assert -0.5 <= median_xi <= 2.0

    def test_to_dataframe(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.parameter_stability_plot()
        df = result.to_dataframe()
        assert "xi" in df.columns
        assert "sigma_star" in df.columns


class TestHillPlot:
    def test_returns_hill_result(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.hill_plot(bootstrap_ci=False)
        assert isinstance(result, HillResult)

    def test_hill_positive_for_pareto_tail(self, pareto_claims):
        claims, threshold, xi_true, sigma = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.hill_plot(bootstrap_ci=False)
        valid = ~np.isnan(result.hill_estimates)
        # Hill estimates should be positive for Pareto-type tail
        assert np.all(result.hill_estimates[valid] > 0)

    def test_hill_estimates_roughly_correct(self, pareto_claims):
        claims, threshold, xi_true, sigma = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.hill_plot(bootstrap_ci=False)
        valid = ~np.isnan(result.hill_estimates)
        # Plateau should be around 1/xi = 1/0.3 ≈ 3.3
        median_h = np.nanmedian(result.hill_estimates[valid])
        # Wide tolerance since we have mixed body+tail data
        assert 0.5 <= median_h <= 8.0

    def test_bootstrap_ci_shape(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.hill_plot(bootstrap_ci=True, n_boot=50)
        assert result.lower_ci is not None
        assert result.upper_ci is not None
        assert len(result.lower_ci) == len(result.hill_estimates)

    def test_to_dataframe(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        result = sel.hill_plot(bootstrap_ci=False)
        df = result.to_dataframe()
        assert "k" in df.columns
        assert "hill_estimate" in df.columns


class TestAutoThreshold:
    def test_returns_float(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        u = sel.auto_threshold()
        assert isinstance(u, float)
        assert u > 0

    def test_in_data_range(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        u = sel.auto_threshold()
        assert u >= claims.min()
        assert u <= claims.max()

    def test_invalid_method(self, pareto_claims):
        claims, *_ = pareto_claims
        sel = ThresholdSelector(claims)
        with pytest.raises(ValueError, match="Unknown method"):
            sel.auto_threshold(method="bader_wadsworth")
