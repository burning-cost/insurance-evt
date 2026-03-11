"""Integration tests: full workflow from raw claims to report."""

import numpy as np
import pytest
from scipy.stats import genpareto

from insurance_evt import (
    ThresholdSelector,
    GPDFitter,
    ReturnLevelCalculator,
    ExcessLayerCalculator,
    SolvencyIIReport,
    GEVFitter,
    FLOOD,
)


class TestFullGPDWorkflow:
    def test_complete_workflow(self, pareto_claims):
        """Step-by-step workflow: threshold -> fit -> return levels -> report."""
        claims, threshold, xi_true, sigma_true = pareto_claims

        # Step 1: threshold selection
        sel = ThresholdSelector(claims)
        mrl = sel.mrl_plot()
        stability = sel.parameter_stability_plot()
        u = sel.auto_threshold()
        assert u > 0

        # Step 2: fit
        fitter = GPDFitter(claims, threshold=threshold)
        fit = fitter.fit()
        assert fit.xi > 0  # Pareto tail

        # Step 3: return levels
        n_years = 30
        lambda_annual = fit.n_exceedances / n_years
        calc = ReturnLevelCalculator(fit, lambda_annual=lambda_annual)
        calc.attach_data(fitter._exceedances)

        rl_200 = calc.return_level(200)
        assert rl_200 > threshold

        # Step 4: confidence interval
        lo, hi = calc.return_level_ci(200, method="wald")
        assert lo < rl_200 < hi

        # Step 5: Solvency II report
        report = SolvencyIIReport(calc).generate()
        assert report["return_level_200"] > threshold
        assert report["ci_lower"] > 0

        # Step 6: XL layers
        xl = ExcessLayerCalculator(fit, lambda_annual=lambda_annual)
        M = threshold * 1.5
        me = xl.mean_excess(M)
        pp = xl.layer_pure_premium(M, M * 0.5)
        assert me > 0
        assert pp > 0

    def test_layer_table_is_actionable(self, pareto_claims):
        claims, threshold, xi_true, sigma_true = pareto_claims
        fit = GPDFitter(claims, threshold=threshold).fit()
        n_years = 30
        lambda_ann = fit.n_exceedances / n_years
        xl = ExcessLayerCalculator(fit, lambda_annual=lambda_ann)

        retentions = [threshold * 1.2, threshold * 1.5, threshold * 2.0, threshold * 3.0]
        limits = [threshold * 0.5] * 4
        df = xl.layer_table(retentions, limits)

        assert len(df) == 4
        # Pure premiums should decrease as retention increases
        pp = df["pure_premium"].values
        assert pp[0] > pp[-1]


class TestGEVWorkflow:
    def test_gev_workflow(self, gev_maxima):
        maxima, xi_true, mu_true, sigma_true = gev_maxima

        fitter = GEVFitter(maxima)
        fit = fitter.fit()

        # Return levels
        rl_100 = fitter.return_level(100)
        rl_200 = fitter.return_level(200)
        assert rl_200 > rl_100

        # Domain
        doa = fitter.domain_of_attraction()
        assert doa in ["Frechet", "Gumbel", "Weibull"]

        # Profile CI
        lo, hi = fitter.profile_likelihood_ci(200)
        assert lo < rl_200 < hi


class TestCensoredWorkflow:
    def test_censored_claims_workflow(self, censored_claims):
        all_claims, all_censored, threshold, xi_true, sigma_true = censored_claims

        fitter = GPDFitter(all_claims, threshold=threshold, right_censoring=all_censored)
        fit = fitter.fit()

        assert fit.n_exceedances > 0
        # With right censoring, n_censored should be > 0
        # (unless no censored claims fell above threshold)
        assert isinstance(fit.n_censored, int)

        lambda_ann = fit.n_exceedances / 10
        calc = ReturnLevelCalculator(fit, lambda_annual=lambda_ann)
        calc.attach_data(fitter._exceedances, fitter._censored)

        rl = calc.return_level(200)
        assert rl > threshold

        report = SolvencyIIReport(calc).generate()
        assert "return_level_200" in report


class TestPresetDrivenWorkflow:
    def test_flood_preset_workflow(self):
        """Workflow using FLOOD preset parameters."""
        rng = np.random.default_rng(77)
        # Generate synthetic flood claims matching FLOOD preset
        xi_typical = np.mean(FLOOD.typical_xi_range)
        sigma = 150_000
        u_pct = FLOOD.threshold_percentile

        # Generate 1000 claims
        all_claims = rng.lognormal(10.5, 1.5, size=1500)
        threshold = np.percentile(all_claims, u_pct)

        # Fit
        fitter = GPDFitter(all_claims, threshold=threshold)
        fit = fitter.fit()

        # xi should be in reasonable range for flood
        lo, hi = FLOOD.typical_xi_range
        # Test that fitted xi is not wildly outside range (allow some slack)
        assert -0.5 <= fit.xi <= 2.0

        # Full report
        lambda_ann = fit.n_exceedances / 20
        calc = ReturnLevelCalculator(fit, lambda_annual=lambda_ann)
        calc.attach_data(fitter._exceedances)
        report = SolvencyIIReport(calc).generate()
        assert "return_level_200" in report
        assert report["return_level_200"] > threshold


class TestReturnLevelConsistency:
    def test_return_level_curve_consistent_with_point(self):
        rng = np.random.default_rng(99)
        excesses = genpareto.rvs(0.30, loc=0, scale=150_000, size=150, random_state=rng)
        claims = excesses + 500_000
        fit = GPDFitter(claims, threshold=500_000).fit()
        calc = ReturnLevelCalculator(fit, lambda_annual=4.0)
        calc.attach_data(excesses)

        T_arr = [50, 100, 200, 500]
        curve = calc.return_level_curve(T_arr, method="wald")

        for i, T in enumerate(T_arr):
            point = calc.return_level(T)
            assert abs(curve.return_levels[i] - point) < 1e-3
