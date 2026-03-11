"""Return level calculation from GPD fit.

Given a fitted GPD and an annual claim arrival rate (lambda), compute the
T-year return level: the claim size expected to be exceeded once every T years.

The formula is:
    x_T = u + (sigma / xi) * [(lambda * T)^xi - 1]    for xi != 0
    x_T = u + sigma * log(lambda * T)                  for xi = 0

where:
    u = threshold
    sigma, xi = GPD parameters
    lambda = annual rate of claims exceeding the threshold
    T = return period in years

For profile likelihood CI on the return level, we reparametrize: holding x_T
fixed, express sigma in terms of (x_T, xi, lambda, T, u), and maximize the
conditional likelihood over xi. This gives asymmetric CIs that correctly
capture the upper tail uncertainty.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .results import GPDFitResult, ReturnLevelCurve
from .gpd import _neg_loglik_standard, _neg_loglik_censored, _neg_loglik_truncated


def _gpd_return_level(
    threshold: float, sigma: float, xi: float, lambda_annual: float, T: float
) -> float:
    """Compute GPD return level for period T years."""
    if T <= 0 or lambda_annual <= 0:
        raise ValueError("T and lambda_annual must be positive.")
    if abs(xi) < 1e-8:
        return threshold + sigma * np.log(lambda_annual * T)
    return threshold + (sigma / xi) * ((lambda_annual * T) ** xi - 1)


class ReturnLevelCalculator:
    """Compute return levels from a GPD fit.

    Parameters
    ----------
    fit : GPDFitResult
        Result from GPDFitter.fit().
    lambda_annual : float
        Annual rate of claims exceeding the threshold. Estimate as
        n_exceedances / n_years.

    Examples
    --------
    >>> fit = GPDFitter(claims, threshold=500_000).fit()
    >>> calc = ReturnLevelCalculator(fit, lambda_annual=3.2)
    >>> calc.return_level(200)
    4_500_000
    >>> curve = calc.return_level_curve([50, 100, 200, 500])
    >>> curve.to_dataframe()
    """

    def __init__(self, fit: GPDFitResult, lambda_annual: float) -> None:
        self.fit = fit
        if lambda_annual <= 0:
            raise ValueError(f"lambda_annual must be positive, got {lambda_annual}")
        self.lambda_annual = float(lambda_annual)
        # Store original excesses for profile likelihood (set externally if needed)
        self._excesses: Optional[np.ndarray] = None
        self._censored: Optional[np.ndarray] = None
        self._trunc_above_threshold: Optional[float] = None

    def attach_data(
        self,
        excesses: np.ndarray,
        censored: Optional[np.ndarray] = None,
        trunc_above_threshold: Optional[float] = None,
    ) -> "ReturnLevelCalculator":
        """Attach raw exceedance data for profile likelihood CI.

        This is called automatically by GPDFitter when constructing a
        ReturnLevelCalculator. You only need it if building the calculator
        manually from a GPDFitResult.
        """
        self._excesses = np.asarray(excesses, dtype=float)
        self._censored = censored
        self._trunc_above_threshold = trunc_above_threshold
        return self

    def return_level(self, return_period: float) -> float:
        """Return level for a given return period.

        Parameters
        ----------
        return_period : float
            In years.

        Returns
        -------
        float
            Claim amount expected to be exceeded once every `return_period` years.
        """
        return _gpd_return_level(
            self.fit.threshold, self.fit.sigma, self.fit.xi, self.lambda_annual, return_period
        )

    def return_level_ci(
        self,
        return_period: float,
        alpha: float = 0.05,
        method: Literal["profile", "wald"] = "profile",
    ) -> tuple:
        """Confidence interval for a return level.

        Parameters
        ----------
        return_period : float
            Return period in years.
        alpha : float
            Significance level (0.05 => 95% CI).
        method : 'profile' or 'wald'
            'profile' gives asymmetric CI via profile log-likelihood — recommended.
            'wald' is faster but symmetric and often too narrow for heavy tails.

        Returns
        -------
        (lower, upper) : tuple of float
        """
        if method == "wald":
            return self._wald_ci(return_period, alpha)
        return self._profile_ci(return_period, alpha)

    def _wald_ci(self, T: float, alpha: float) -> tuple:
        """Wald CI using delta method."""
        fit = self.fit
        z = stats.norm.ppf(1 - alpha / 2)
        x_T = self.return_level(T)
        # Delta method: approximate SE
        lT = self.lambda_annual * T
        if abs(fit.xi) < 1e-8:
            dxT_dxi = fit.sigma * np.log(lT) ** 2
            dxT_dsigma = np.log(lT)
        else:
            dxT_dxi = (
                fit.sigma / fit.xi**2 * ((lT**fit.xi) * (fit.xi * np.log(lT) - 1) + 1)
            )
            dxT_dsigma = (lT**fit.xi - 1) / fit.xi
        se_xT = np.sqrt((dxT_dxi * fit.xi_se) ** 2 + (dxT_dsigma * fit.sigma_se) ** 2)
        return (x_T - z * se_xT, x_T + z * se_xT)

    def _profile_ci(self, T: float, alpha: float) -> tuple:
        """Profile likelihood CI for return level x_T.

        Reparametrize: given x_T, express sigma as
            sigma = xi * (x_T - u) / [(lambda*T)^xi - 1]  for xi != 0
        and maximize the profile log-likelihood over xi.
        """
        if self._excesses is None:
            # Fall back to Wald if no data attached
            return self._wald_ci(T, alpha)

        fit = self.fit
        excesses = self._excesses
        censored = self._censored
        trunc = self._trunc_above_threshold

        if trunc is not None:
            def neg_ll_fn(p):
                return _neg_loglik_truncated(p, excesses, trunc)
        elif censored is not None and np.any(censored):
            def neg_ll_fn(p):
                return _neg_loglik_censored(p, excesses, censored)
        else:
            def neg_ll_fn(p):
                return _neg_loglik_standard(p, excesses)

        loglik_max = fit.loglik
        cutoff = loglik_max - stats.chi2.ppf(1 - alpha, df=1) / 2
        x_T_mle = self.return_level(T)

        def sigma_from_xT_xi(x_T, xi):
            """Express sigma as function of (x_T, xi) given return level equation."""
            lT = self.lambda_annual * T
            delta = x_T - fit.threshold
            if abs(xi) < 1e-8:
                return delta / np.log(lT)
            denom = lT**xi - 1
            if abs(denom) < 1e-12:
                return delta / np.log(lT)
            return xi * delta / denom

        def profile_loglik_xT(x_T):
            """Profile log-likelihood at fixed x_T, maximized over xi."""
            def neg_ll_1d(xi_arr):
                xi_ = xi_arr[0]
                sigma_ = sigma_from_xT_xi(x_T, xi_)
                if sigma_ <= 0:
                    return 1e12
                return neg_ll_fn([xi_, sigma_])

            r = optimize.minimize(
                neg_ll_1d,
                [fit.xi],
                method="L-BFGS-B",
                bounds=[(-0.99, 10.0)],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            return -r.fun

        def objective(x_T):
            return profile_loglik_xT(x_T) - cutoff

        # Search bounds: start with ±wide margin
        scale = max(abs(x_T_mle) * 0.5, fit.sigma * 5)
        lo_start = max(fit.threshold, x_T_mle - scale)
        hi_start = x_T_mle + scale * 3

        try:
            lo = optimize.brentq(objective, lo_start, x_T_mle - 1, xtol=abs(x_T_mle) * 1e-4, maxiter=100)
        except ValueError:
            lo = lo_start

        try:
            hi = optimize.brentq(objective, x_T_mle + 1, hi_start, xtol=abs(x_T_mle) * 1e-4, maxiter=100)
        except ValueError:
            hi = hi_start

        return (float(lo), float(hi))

    def return_level_curve(
        self,
        return_periods: Union[Sequence[float], np.ndarray],
        alpha: float = 0.05,
        method: Literal["profile", "wald"] = "profile",
    ) -> ReturnLevelCurve:
        """Return level curve with CIs across multiple return periods.

        Parameters
        ----------
        return_periods : array-like
            Return periods in years to evaluate.
        alpha : float
            Confidence level.
        method : str
            'profile' (recommended) or 'wald'.

        Returns
        -------
        ReturnLevelCurve
        """
        T_arr = np.asarray(return_periods, dtype=float)
        rl = np.array([self.return_level(T) for T in T_arr])
        lo = np.zeros(len(T_arr))
        hi = np.zeros(len(T_arr))
        for i, T in enumerate(T_arr):
            lo[i], hi[i] = self.return_level_ci(T, alpha=alpha, method=method)
        return ReturnLevelCurve(
            return_periods=T_arr,
            return_levels=rl,
            lower_ci=lo,
            upper_ci=hi,
            alpha=alpha,
        )
