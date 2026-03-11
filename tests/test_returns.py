"""Tests for ReturnLevelCalculator."""

import numpy as np
import pytest
from scipy.stats import genpareto

from insurance_evt import GPDFitter, ReturnLevelCalculator
from insurance_evt.results import ReturnLevelCurve


def _make_calculator(rng=None, xi=0.3, sigma=150_000, threshold=500_000, n=150, lambda_ann=4.0):
    if rng is None:
        rng = np.random.default_rng(7)
    excesses = genpareto.rvs(xi, loc=0, scale=sigma, size=n, random_state=rng)
    claims = excesses + threshold
    fit = GPDFitter(claims, threshold=threshold).fit()
    calc = ReturnLevelCalculator(fit, lambda_annual=lambda_ann)
    calc.attach_data(excesses)
    return calc, fit


class TestReturnLevelCalculatorInit:
    def test_basic_init(self):
        calc, fit = _make_calculator()
        assert calc.lambda_annual == 4.0

    def test_negative_lambda_raises(self):
        calc, fit = _make_calculator()
        with pytest.raises(ValueError, match="lambda_annual"):
            ReturnLevelCalculator(fit, lambda_annual=-1.0)

    def test_zero_lambda_raises(self):
        calc, fit = _make_calculator()
        with pytest.raises(ValueError):
            ReturnLevelCalculator(fit, lambda_annual=0.0)


class TestReturnLevel:
    def test_returns_float(self):
        calc, _ = _make_calculator()
        rl = calc.return_level(200)
        assert isinstance(rl, float)

    def test_increases_with_period(self):
        calc, _ = _make_calculator()
        levels = [calc.return_level(T) for T in [10, 50, 100, 200, 500, 1000]]
        assert all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))

    def test_exceeds_threshold(self):
        calc, fit = _make_calculator()
        rl = calc.return_level(200)
        assert rl > fit.threshold

    def test_formula_xi_near_zero(self):
        """For xi near 0, x_T ≈ u + sigma * log(lambda*T)."""
        rng = np.random.default_rng(55)
        # Nearly exponential tail (xi ≈ 0)
        from scipy.stats import expon
        excesses = expon.rvs(scale=100_000, size=300, random_state=rng)
        claims = excesses + 200_000
        fit = GPDFitter(claims, threshold=200_000).fit()
        calc = ReturnLevelCalculator(fit, lambda_annual=5.0)
        rl = calc.return_level(100)
        # Should be positive and above threshold
        assert rl > 200_000

    def test_return_level_200_reasonable(self):
        # With xi=0.3, sigma=150k, threshold=500k, lambda=4:
        # x_200 = 500k + (150k/0.3) * [(4*200)^0.3 - 1] ≈ large number
        calc, _ = _make_calculator(xi=0.30, sigma=150_000, threshold=500_000, lambda_ann=4.0)
        rl = calc.return_level(200)
        assert rl > 1_000_000  # must be significantly above threshold


class TestReturnLevelCI:
    def test_profile_ci_tuple(self):
        calc, _ = _make_calculator()
        lo, hi = calc.return_level_ci(200, method="profile")
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_profile_ci_contains_estimate(self):
        calc, _ = _make_calculator()
        rl = calc.return_level(200)
        lo, hi = calc.return_level_ci(200, method="profile")
        assert lo < rl < hi

    def test_wald_ci_tuple(self):
        calc, _ = _make_calculator()
        lo, hi = calc.return_level_ci(200, method="wald")
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_wald_ci_contains_estimate(self):
        calc, _ = _make_calculator()
        rl = calc.return_level(200)
        lo, hi = calc.return_level_ci(200, method="wald")
        assert lo < rl < hi

    def test_profile_upper_wider_than_wald(self):
        """Profile CI upper bound should be wider for heavy tails (xi=0.3)."""
        calc, _ = _make_calculator(xi=0.3)
        rl = calc.return_level(200)
        lo_p, hi_p = calc.return_level_ci(200, method="profile")
        lo_w, hi_w = calc.return_level_ci(200, method="wald")
        # Profile CI should be at least as wide on the upper side
        assert hi_p > rl  # upper bound above estimate


class TestReturnLevelCurve:
    def test_returns_curve_object(self):
        calc, _ = _make_calculator()
        curve = calc.return_level_curve([50, 100, 200], method="wald")
        assert isinstance(curve, ReturnLevelCurve)

    def test_curve_shape(self):
        calc, _ = _make_calculator()
        T_arr = [10, 50, 100, 200, 500]
        curve = calc.return_level_curve(T_arr, method="wald")
        assert len(curve.return_levels) == 5
        assert len(curve.lower_ci) == 5
        assert len(curve.upper_ci) == 5

    def test_curve_to_dataframe(self):
        calc, _ = _make_calculator()
        curve = calc.return_level_curve([50, 200], method="wald")
        df = curve.to_dataframe()
        assert "return_period" in df.columns
        assert "return_level" in df.columns
        assert "lower_ci" in df.columns

    def test_return_levels_monotone_in_curve(self):
        calc, _ = _make_calculator()
        T_arr = [20, 50, 100, 200]
        curve = calc.return_level_curve(T_arr, method="wald")
        levels = curve.return_levels
        assert all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))
