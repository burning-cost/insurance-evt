"""Solvency II SCR reporting from GPD/GEV fits.

Solvency II requires insurers using internal models to estimate the 1-in-200
year loss (99.5th percentile, Value at Risk). For catastrophe perils, this is
the key output of EVT analysis.

This module wraps ReturnLevelCalculator to produce the structured output that
an internal model team needs: the point estimate, asymmetric profile likelihood
CI, key model parameters, and a plain-English interpretation suitable for
inclusion in model documentation or regulatory submissions.

Note on non-stationarity:
    Standard EVT assumes the claims are iid. For UK flood, subsidence, and
    related weather-driven perils, this is increasingly questionable due to
    climate trends. Solvency II internal model teams should document this
    assumption explicitly. A future version of this library will support
    non-stationary GEV; for now, include the warning in all reports.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .results import GPDFitResult
from .returns import ReturnLevelCalculator


# Solvency II standard return period
SOLVENCY_II_RETURN_PERIOD = 200


class SolvencyIIReport:
    """1-in-200 year return level report for Solvency II internal models.

    Parameters
    ----------
    calculator : ReturnLevelCalculator
        Fitted and configured return level calculator.

    Examples
    --------
    >>> fit = GPDFitter(claims, threshold=500_000).fit()
    >>> calc = ReturnLevelCalculator(fit, lambda_annual=3.5)
    >>> report = SolvencyIIReport(calc)
    >>> result = report.generate()
    >>> print(result['interpretation'])
    'The 1-in-200 year claim is estimated at £4.2m (95% CI: £2.8m to £7.6m)...'
    """

    def __init__(self, calculator: ReturnLevelCalculator) -> None:
        self.calculator = calculator
        self._result: Optional[dict] = None

    def generate(self, alpha_ci: float = 0.05) -> dict:
        """Generate the Solvency II SCR estimate.

        Parameters
        ----------
        alpha_ci : float
            Confidence level for the profile likelihood CI (default 0.05 => 95%).

        Returns
        -------
        dict with keys:
            return_level_200 : float
                Point estimate of 1-in-200 year claim.
            ci_lower : float
                Lower bound of profile likelihood CI.
            ci_upper : float
                Upper bound of profile likelihood CI.
            ci_alpha : float
                Significance level used.
            xi : float
                GPD shape parameter.
            xi_se : float
                Standard error of xi.
            sigma : float
                GPD scale parameter.
            threshold : float
                Threshold u.
            lambda_annual : float
                Annual exceedance rate.
            n_exceedances : int
                Sample size above threshold.
            n_censored : int
                Number of right-censored observations.
            interpretation : str
                Plain-English description for model documentation.
            warnings : list of str
                Actuarial warnings (small sample, high xi, censoring, etc.).
        """
        fit = self.calculator.fit
        T = SOLVENCY_II_RETURN_PERIOD

        rl_200 = self.calculator.return_level(T)
        ci_lower, ci_upper = self.calculator.return_level_ci(T, alpha=alpha_ci, method="profile")

        warnings = []

        # Sample size warning
        if fit.n_exceedances < 30:
            warnings.append(
                f"Small sample: only {fit.n_exceedances} exceedances above threshold. "
                "Profile likelihood CI may be very wide. Consider lower threshold or "
                "pooling with similar perils."
            )
        elif fit.n_exceedances < 50:
            warnings.append(
                f"Moderate sample: {fit.n_exceedances} exceedances. "
                "CIs will be material. Do not round the point estimate."
            )

        # Heavy tail warning
        if fit.xi > 0.5:
            warnings.append(
                f"Heavy tail: xi = {fit.xi:.3f} (> 0.5). "
                "Return levels are highly sensitive to xi. "
                "Profile CI upper bound is the more conservative capital figure."
            )

        # Censoring warning
        if fit.n_censored > 0:
            cens_pct = 100 * fit.n_censored / fit.n_exceedances
            warnings.append(
                f"Censoring: {fit.n_censored} of {fit.n_exceedances} exceedances ({cens_pct:.0f}%) "
                "are right-censored (open claims). Censored MLE applied. "
                "Final development of open claims could move xi."
            )

        # Stationarity warning (always)
        warnings.append(
            "Stationarity assumed: standard EVT requires iid claims. "
            "For weather-driven perils (flood, subsidence, storm), climate trends "
            "may violate this assumption. Document and review annually."
        )

        # High xi / infinite mean warning
        if fit.xi >= 1:
            warnings.append(
                f"CRITICAL: xi = {fit.xi:.3f} >= 1. Infinite mean — return level "
                "formula requires xi < 1. Results unreliable. Investigate data quality."
            )

        # Format interpretation
        def fmt(x: float) -> str:
            if x >= 1e6:
                return f"£{x/1e6:.1f}m"
            elif x >= 1e3:
                return f"£{x/1e3:.0f}k"
            else:
                return f"£{x:.0f}"

        pct = int(100 * (1 - alpha_ci))
        interpretation = (
            f"The 1-in-{T} year claim severity is estimated at {fmt(rl_200)} "
            f"({pct}% CI: {fmt(ci_lower)} to {fmt(ci_upper)}, "
            f"profile likelihood). "
            f"Fitted GPD: xi = {fit.xi:.3f} (shape), sigma = {fmt(fit.sigma)} (scale), "
            f"threshold = {fmt(fit.threshold)}, "
            f"based on {fit.n_exceedances} exceedances "
            f"at annual rate {self.calculator.lambda_annual:.2f}/year."
        )

        self._result = {
            "return_level_200": float(rl_200),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_alpha": float(alpha_ci),
            "xi": float(fit.xi),
            "xi_se": float(fit.xi_se),
            "sigma": float(fit.sigma),
            "sigma_se": float(fit.sigma_se),
            "threshold": float(fit.threshold),
            "lambda_annual": float(self.calculator.lambda_annual),
            "n_exceedances": int(fit.n_exceedances),
            "n_censored": int(fit.n_censored),
            "interpretation": interpretation,
            "warnings": warnings,
        }
        return self._result

    def to_dataframe(self) -> pd.DataFrame:
        """Return the report as a single-row DataFrame.

        Useful for logging or combining multiple peril reports.
        """
        if self._result is None:
            self.generate()
        r = self._result
        return pd.DataFrame(
            [
                {
                    "return_level_200": r["return_level_200"],
                    "ci_lower_95": r["ci_lower"],
                    "ci_upper_95": r["ci_upper"],
                    "xi": r["xi"],
                    "xi_se": r["xi_se"],
                    "sigma": r["sigma"],
                    "threshold": r["threshold"],
                    "lambda_annual": r["lambda_annual"],
                    "n_exceedances": r["n_exceedances"],
                    "n_censored": r["n_censored"],
                }
            ]
        )
