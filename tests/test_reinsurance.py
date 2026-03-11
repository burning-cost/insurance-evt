"""Tests for ExcessLayerCalculator."""

import numpy as np
import pytest
from scipy.stats import genpareto

from insurance_evt import GPDFitter, ExcessLayerCalculator
from insurance_evt.results import GPDFitResult


def _make_calculator(xi=0.3, sigma=200_000, threshold=500_000, lambda_ann=3.0, n=200):
    rng = np.random.default_rng(13)
    excesses = genpareto.rvs(xi, loc=0, scale=sigma, size=n, random_state=rng)
    claims = excesses + threshold
    fit = GPDFitter(claims, threshold=threshold).fit()
    return ExcessLayerCalculator(fit, lambda_annual=lambda_ann)


class TestExcessLayerCalculatorInit:
    def test_init(self):
        calc = _make_calculator()
        assert calc.lambda_annual == 3.0

    def test_zero_lambda_raises(self):
        calc = _make_calculator()
        from insurance_evt.results import GPDFitResult
        fit = calc.fit
        with pytest.raises(ValueError):
            ExcessLayerCalculator(fit, lambda_annual=0.0)


class TestMeanExcess:
    def test_returns_float(self):
        calc = _make_calculator()
        me = calc.mean_excess(600_000)
        assert isinstance(me, float)

    def test_positive(self):
        calc = _make_calculator()
        assert calc.mean_excess(600_000) > 0

    def test_increases_with_retention(self):
        """E[X - M | X > M] should increase with M for xi > 0 (Pareto tail)."""
        calc = _make_calculator(xi=0.3)
        levels = [600_000, 800_000, 1_000_000, 1_500_000]
        excesses = [calc.mean_excess(M) for M in levels]
        # Mean excess is linear in M for GPD => should increase
        assert all(excesses[i] < excesses[i + 1] for i in range(len(excesses) - 1))

    def test_formula_verification(self):
        """E[X-M|X>M] = sigma_M / (1-xi) where sigma_M = sigma + xi*(M-u)."""
        # Use a fixed GPDFitResult to verify the formula directly
        fit = GPDFitResult(
            xi=0.30,
            sigma=200_000,
            xi_se=0.05,
            sigma_se=20_000,
            n_exceedances=100,
            loglik=-500.0,
            threshold=500_000,
        )
        calc = ExcessLayerCalculator(fit, lambda_annual=3.0)
        M = 700_000
        sigma_M = 200_000 + 0.30 * (700_000 - 500_000)
        expected = sigma_M / (1 - 0.30)
        assert abs(calc.mean_excess(M) - expected) < 1.0

    def test_retention_below_threshold_raises(self):
        calc = _make_calculator(threshold=500_000)
        with pytest.raises(ValueError, match="threshold"):
            calc.mean_excess(400_000)

    def test_xi_ge_1_raises(self):
        fit = GPDFitResult(
            xi=1.1,
            sigma=100_000,
            xi_se=0.1,
            sigma_se=10_000,
            n_exceedances=50,
            loglik=-200.0,
            threshold=200_000,
        )
        calc = ExcessLayerCalculator(fit, lambda_annual=2.0)
        with pytest.raises(ValueError, match="xi.*>= 1"):
            calc.mean_excess(300_000)


class TestLayerPurePremium:
    def test_returns_float(self):
        calc = _make_calculator()
        pp = calc.layer_pure_premium(600_000, 400_000)
        assert isinstance(pp, float)

    def test_unlimited_greater_than_capped(self):
        calc = _make_calculator()
        M = 600_000
        pp_capped = calc.layer_pure_premium(M, 500_000)
        pp_unlimited = calc.layer_pure_premium(M, None)
        assert pp_unlimited > pp_capped

    def test_positive_premium(self):
        calc = _make_calculator()
        assert calc.layer_pure_premium(600_000, 400_000) > 0

    def test_premium_decreases_with_retention(self):
        """Higher retention => lower pure premium (reinsurer pays less)."""
        calc = _make_calculator()
        limit = 500_000
        pp_low = calc.layer_pure_premium(600_000, limit)
        pp_high = calc.layer_pure_premium(900_000, limit)
        assert pp_low > pp_high

    def test_formula_unlimited(self):
        """Unlimited PP = lambda * integral_a^inf [1-F(y)] dy."""
        # Direct verification against formula from KB
        fit = GPDFitResult(
            xi=0.30,
            sigma=200_000,
            xi_se=0.05,
            sigma_se=20_000,
            n_exceedances=100,
            loglik=-500.0,
            threshold=500_000,
        )
        calc = ExcessLayerCalculator(fit, lambda_annual=3.0)
        # Unlimited XL above 700k: a = 200k above threshold
        a = 200_000
        xi, sigma = 0.30, 200_000
        # integral = sigma/(1-xi) * (1 + xi*a/sigma)^(1-1/xi)
        expected_integral = sigma / (1 - xi) * (1 + xi * a / sigma) ** (1 - 1 / xi)
        pp = calc.layer_pure_premium(700_000, None)
        expected_pp = 3.0 * expected_integral
        assert abs(pp - expected_pp) / expected_pp < 0.01

    def test_retention_below_threshold_raises(self):
        calc = _make_calculator(threshold=500_000)
        with pytest.raises(ValueError, match="threshold"):
            calc.layer_pure_premium(400_000)


class TestLayerTable:
    def test_returns_dataframe(self):
        import pandas as pd
        calc = _make_calculator()
        df = calc.layer_table([600_000, 800_000, 1_000_000], [400_000, 500_000, 500_000])
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        calc = _make_calculator()
        df = calc.layer_table([600_000, 800_000], [400_000, 500_000])
        assert len(df) == 2

    def test_columns(self):
        calc = _make_calculator()
        df = calc.layer_table([700_000], [300_000])
        assert "retention" in df.columns
        assert "limit" in df.columns
        assert "pure_premium" in df.columns
        assert "mean_excess" in df.columns
        assert "rate_on_line" in df.columns

    def test_rate_on_line_positive(self):
        calc = _make_calculator()
        df = calc.layer_table([600_000, 800_000], [300_000, 400_000])
        assert (df["rate_on_line"] > 0).all()

    def test_rate_on_line_definition(self):
        """Rate on line = PP / limit."""
        calc = _make_calculator()
        df = calc.layer_table([700_000], [400_000])
        expected = df["pure_premium"].iloc[0] / df["limit"].iloc[0]
        assert abs(df["rate_on_line"].iloc[0] - expected) < 1e-6

    def test_unlimited_layer(self):
        calc = _make_calculator()
        df = calc.layer_table([700_000], [None])
        assert np.isinf(df["limit"].iloc[0])

    def test_pure_premiums_decreasing_in_retention(self):
        calc = _make_calculator()
        df = calc.layer_table(
            [600_000, 700_000, 800_000, 900_000],
            [300_000, 300_000, 300_000, 300_000]
        )
        pp = df["pure_premium"].values
        assert all(pp[i] >= pp[i + 1] for i in range(len(pp) - 1))

    def test_wrong_lengths_raises(self):
        calc = _make_calculator()
        with pytest.raises(ValueError, match="same length"):
            calc.layer_table([600_000, 700_000], [300_000])
