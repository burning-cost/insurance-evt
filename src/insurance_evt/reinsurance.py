"""Excess layer pure premium calculation for reinsurance pricing.

This is the actuarial pricing output that pricing teams actually need.
The mean excess function and layer pure premium translate GPD parameters
into the annual expected cost of each XL layer.

This is the first Python implementation of the ExcessGPD formula from the
ReIns R package (Reynkens et al., 2017). R users have had this for years.
Now Python reinsurance teams do too.

Key formula (from KB entry 1376):
For GPD(xi, sigma_u) fitted above threshold u, and XL layer from M to M+limit:

    sigma_M = sigma_u + xi * (M - u)       [scale at retention]
    e(M) = sigma_M / (1 - xi)              [mean excess at M, requires xi < 1]

    Layer pure premium (unlimited above M):
        PP = lambda * sigma_M / (1 - xi)

    Layer pure premium (M xs M+limit):
        PP = lambda * integral_{M-u}^{M-u+limit} [1 - F_GPD(y)] dy

Closed form for the integral:
    For xi != 0:
        integral_a^b [1 - F(y; xi, sigma)] dy
            = sigma/(1-xi) * [(1 + xi*a/sigma)^(1-1/xi) - (1 + xi*b/sigma)^(1-1/xi)] / xi
    For xi = 0:
        integral_a^b [1 - F(y; 0, sigma)] dy = sigma * [exp(-a/sigma) - exp(-b/sigma)]
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .results import GPDFitResult


def _gpd_integral_survival(
    a: float, b: Optional[float], xi: float, sigma: float
) -> float:
    """Integral of GPD survival function from a to b (or infinity if b is None).

    integral_a^b [1 - F(y; xi, sigma)] dy

    For insurance: a = M - u (retention minus threshold), b = M + limit - u.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if abs(xi) < 1e-8:
        # Exponential limit
        result = sigma * np.exp(-a / sigma)
        if b is not None:
            result -= sigma * np.exp(-b / sigma)
        return result

    # GPD case
    if xi > 0:
        # Pareto-type: support is [0, infinity)
        def S_pow(y):
            return (1 + xi * y / sigma) ** (1 - 1 / xi)

        integral_from_a = sigma / (1 - xi) * S_pow(a)
        if b is None:
            return integral_from_a
        # Check b is within support
        if 1 + xi * b / sigma <= 0:
            return integral_from_a  # upper bound outside support
        return integral_from_a - sigma / (1 - xi) * S_pow(b)
    else:
        # xi < 0: bounded support at -sigma/xi
        max_excess = -sigma / xi
        a = min(a, max_excess)

        def S_pow(y):
            arg = 1 + xi * y / sigma
            if arg <= 0:
                return 0.0
            return arg ** (1 - 1 / xi)

        integral_from_a = sigma / (1 - xi) * S_pow(a)
        if b is None:
            return integral_from_a
        b_eff = min(b, max_excess)
        return integral_from_a - sigma / (1 - xi) * S_pow(b_eff)


class ExcessLayerCalculator:
    """Excess layer pure premium from GPD parameters.

    Parameters
    ----------
    fit : GPDFitResult
        GPD fit result from GPDFitter.fit().
    lambda_annual : float
        Annual rate of claims exceeding the GPD threshold.
        Estimate as n_exceedances / n_years.

    Notes
    -----
    All methods require xi < 1. For xi >= 1, the mean excess is infinite and
    reinsurance pricing is theoretically undefined. In practice, if your MLE
    gives xi > 0.8, widen the confidence interval — the true xi may be < 1.
    The profile_likelihood_ci on xi can confirm this.
    """

    def __init__(self, fit: GPDFitResult, lambda_annual: float) -> None:
        self.fit = fit
        if lambda_annual <= 0:
            raise ValueError(f"lambda_annual must be positive, got {lambda_annual}")
        self.lambda_annual = float(lambda_annual)

    def _check_xi(self) -> None:
        if self.fit.xi >= 1:
            raise ValueError(
                f"xi = {self.fit.xi:.3f} >= 1. Mean excess is infinite; "
                "excess layer pricing undefined. "
                "Check profile likelihood CI — if upper bound < 1, proceed with caution."
            )

    def _sigma_at_retention(self, retention: float) -> float:
        """Effective GPD scale at retention M: sigma_M = sigma + xi * (M - u)."""
        return self.fit.sigma + self.fit.xi * (retention - self.fit.threshold)

    def mean_excess(self, retention: float) -> float:
        """Expected excess above retention M, given claim exceeds M.

        E[X - M | X > M] = sigma_M / (1 - xi)

        where sigma_M = sigma + xi * (M - u) is the modified scale at M.

        Parameters
        ----------
        retention : float
            Retention (attachment) point M. Must be >= threshold.

        Returns
        -------
        float
            Expected claim excess above M per occurrence that exceeds M.
        """
        self._check_xi()
        if retention < self.fit.threshold:
            raise ValueError(
                f"retention ({retention}) must be >= threshold ({self.fit.threshold})"
            )
        sigma_M = self._sigma_at_retention(retention)
        if sigma_M <= 0:
            raise ValueError(
                f"Negative effective scale sigma_M = {sigma_M:.0f} at retention {retention}. "
                "GPD model breaks down at this retention; use a lower threshold."
            )
        return sigma_M / (1 - self.fit.xi)

    def layer_pure_premium(
        self, retention: float, limit: Optional[float] = None
    ) -> float:
        """Annual expected cost of an XL layer.

        For unlimited cover above retention M:
            PP = lambda * P(X > M) * E[X - M | X > M]
               = lambda * S_GPD(M - u) * sigma_M / (1 - xi)

        For capped layer M xs M+limit:
            PP = lambda * integral_{M-u}^{M-u+limit} S_GPD(y) dy

        Parameters
        ----------
        retention : float
            Retention (attachment) M. Claims below this are not covered.
        limit : float, optional
            Layer limit. If None, cover is unlimited above retention.

        Returns
        -------
        float
            Annual expected layer cost (pure premium, before expense loading).
        """
        self._check_xi()
        if retention < self.fit.threshold:
            raise ValueError(
                f"retention ({retention}) must be >= threshold ({self.fit.threshold})"
            )

        fit = self.fit
        u = fit.threshold
        xi = fit.xi
        sigma = fit.sigma

        a = retention - u  # excess above threshold at retention
        b = None if limit is None else a + limit

        # Check valid support for xi < 0
        if xi < 0:
            max_excess = -sigma / xi
            if a >= max_excess:
                return 0.0  # retention above GPD support

        integral = _gpd_integral_survival(a, b, xi, sigma)
        return self.lambda_annual * integral

    def layer_table(
        self,
        retentions: Sequence[float],
        limits: Sequence[Optional[float]],
    ) -> pd.DataFrame:
        """Compute pure premiums for a grid of retentions and limits.

        Parameters
        ----------
        retentions : sequence of float
            Retention levels to evaluate.
        limits : sequence of float or None
            Corresponding limits. Use None for unlimited covers.
            If a single value, applied to all retentions.

        Returns
        -------
        pd.DataFrame
            Columns: retention, limit, mean_excess_at_retention, pure_premium,
                     rate_on_line (pp / limit, for capped layers)
        """
        retentions = list(retentions)
        if not hasattr(limits, "__len__") or isinstance(limits, (float, int)):
            limits = [limits] * len(retentions)
        else:
            limits = list(limits)

        if len(limits) != len(retentions):
            raise ValueError(
                f"retentions ({len(retentions)}) and limits ({len(limits)}) must have same length."
            )

        rows = []
        for M, lim in zip(retentions, limits):
            try:
                me = self.mean_excess(M)
            except Exception:
                me = np.nan
            try:
                pp = self.layer_pure_premium(M, lim)
            except Exception:
                pp = np.nan
            rol = pp / lim if (lim is not None and lim > 0 and np.isfinite(pp)) else np.nan
            rows.append(
                {
                    "retention": M,
                    "limit": lim if lim is not None else np.inf,
                    "mean_excess": me,
                    "pure_premium": pp,
                    "rate_on_line": rol,
                }
            )
        return pd.DataFrame(rows)
