"""Threshold selection diagnostics for GPD fitting.

The right threshold choice is the single most consequential decision in EVT.
Too low: GPD approximation not valid, bias in parameter estimates.
Too high: too few exceedances, variance blows up.

This module provides diagnostic plots (MRL, parameter stability, Hill estimator)
and a simple automated suggestion. The diagnostics are the product — the
auto_threshold is a starting point, not the answer.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .results import HillResult, MRLResult, StabilityResult


def _fit_gpd_fast(excesses: np.ndarray) -> Tuple[float, float, float, float]:
    """Quick GPD MLE. Returns (xi, sigma, xi_se, sigma_se).

    Uses scipy internally. Falls back to method-of-moments if MLE fails.
    """
    from scipy.stats import genpareto
    from scipy.optimize import minimize

    n = len(excesses)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan

    # scipy.stats.genpareto uses (c, loc, scale) = (xi, 0, sigma)
    try:
        xi, loc, sigma = genpareto.fit(excesses, floc=0)
        # Standard errors from observed Fisher information (numerical Hessian)
        def neg_loglik(params):
            xi_, sigma_ = params
            if sigma_ <= 0:
                return 1e10
            return -np.sum(genpareto.logpdf(excesses, xi_, loc=0, scale=sigma_))

        result = minimize(
            neg_loglik, [xi, sigma], method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000}
        )
        xi, sigma = result.x
        # Numerical Hessian for SE
        from scipy.optimize import approx_fprime
        h = 1e-5
        def grad(p):
            return approx_fprime(p, neg_loglik, h)
        hess = np.zeros((2, 2))
        for i in range(2):
            ei = np.zeros(2)
            ei[i] = h
            hess[:, i] = (grad(result.x + ei) - grad(result.x - ei)) / (2 * h)
        try:
            cov = np.linalg.inv(hess)
            xi_se = np.sqrt(max(cov[0, 0], 0))
            sigma_se = np.sqrt(max(cov[1, 1], 0))
        except np.linalg.LinAlgError:
            xi_se = np.abs(xi) / np.sqrt(n)
            sigma_se = sigma / np.sqrt(n)
        return xi, sigma, xi_se, sigma_se
    except Exception:
        # Moment estimators as fallback
        m1 = np.mean(excesses)
        m2 = np.mean(excesses**2)
        if m2 > 2 * m1**2:
            xi = 0.5 * (1 - m1**2 / (m2 / 2 - m1**2))
            sigma = m1 * (1 - xi)
        else:
            xi, sigma = 0.0, m1
        xi_se = np.abs(xi) / np.sqrt(n) if n > 0 else np.nan
        sigma_se = sigma / np.sqrt(n) if n > 0 else np.nan
        return xi, sigma, xi_se, sigma_se


class ThresholdSelector:
    """Diagnostic tools for GPD threshold selection.

    The standard workflow is:

    1. hill_plot()  — is the tail Pareto-type at all?
    2. mrl_plot()   — where does mean residual life become linear?
    3. parameter_stability_plot() — where do GPD parameters stabilise?
    4. auto_threshold() — algorithmic suggestion (sanity check, not verdict)

    The diagnostics should guide your judgement, not replace it. Two analysts
    looking at the same plots may reasonably pick thresholds that differ by
    10-15%. That's fine — use profile_likelihood_ci on your GPD fit to quantify
    the resulting uncertainty.

    Parameters
    ----------
    claims : array-like
        Raw claim amounts. All positive values. No need to pre-filter by
        threshold — the methods do that internally.
    """

    def __init__(self, claims: np.ndarray) -> None:
        self.claims = np.asarray(claims, dtype=float)
        if np.any(self.claims <= 0):
            raise ValueError("All claims must be strictly positive.")
        self._sorted = np.sort(self.claims)[::-1]  # descending

    def _default_thresholds(self, n_points: int = 30) -> np.ndarray:
        """Log-spaced thresholds from 50th to 95th percentile."""
        lo = np.percentile(self.claims, 50)
        hi = np.percentile(self.claims, 95)
        return np.exp(np.linspace(np.log(lo), np.log(hi), n_points))

    def mrl_plot(
        self,
        thresholds: Optional[np.ndarray] = None,
        alpha: float = 0.95,
        ax=None,
    ) -> MRLResult:
        """Mean Residual Life (mean excess) plot.

        Under the GPD approximation, E[X - u | X > u] should be linear in u.
        The threshold should be chosen where this linearity begins.

        Parameters
        ----------
        thresholds : array-like, optional
            Threshold values to evaluate. Defaults to 30 log-spaced points
            between the 50th and 95th percentile.
        alpha : float
            Confidence level for pointwise CIs (based on CLT).
        ax : matplotlib Axes, optional
            If provided, plot to this axes.

        Returns
        -------
        MRLResult
        """
        if thresholds is None:
            thresholds = self._default_thresholds()
        thresholds = np.asarray(thresholds, dtype=float)

        mean_excess = np.full(len(thresholds), np.nan)
        lower_ci = np.full(len(thresholds), np.nan)
        upper_ci = np.full(len(thresholds), np.nan)
        n_above = np.zeros(len(thresholds), dtype=int)

        z = stats.norm.ppf(0.5 + alpha / 2)

        for i, u in enumerate(thresholds):
            excesses = self.claims[self.claims > u] - u
            n = len(excesses)
            n_above[i] = n
            if n < 2:
                continue
            me = np.mean(excesses)
            se = np.std(excesses, ddof=1) / np.sqrt(n)
            mean_excess[i] = me
            lower_ci[i] = me - z * se
            upper_ci[i] = me + z * se

        if ax is not None:
            self._plot_mrl(ax, thresholds, mean_excess, lower_ci, upper_ci, n_above)

        return MRLResult(
            thresholds=thresholds,
            mean_excess=mean_excess,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            n_above=n_above,
        )

    def _plot_mrl(self, ax, thresholds, mean_excess, lower_ci, upper_ci, n_above):
        valid = ~np.isnan(mean_excess)
        ax.plot(thresholds[valid], mean_excess[valid], "b-o", markersize=3, label="Mean excess")
        ax.fill_between(
            thresholds[valid], lower_ci[valid], upper_ci[valid], alpha=0.2, color="blue"
        )
        ax.set_xlabel("Threshold u")
        ax.set_ylabel("E[X − u | X > u]")
        ax.set_title("Mean Residual Life Plot\n(choose threshold where line becomes linear)")
        ax.legend()

    def parameter_stability_plot(
        self,
        thresholds: Optional[np.ndarray] = None,
        ax=None,
    ) -> StabilityResult:
        """GPD parameter stability plot.

        Fits GPD at each threshold and plots xi and the modified scale
        sigma* = sigma - xi * u (which should be constant across thresholds if
        the GPD is correctly specified).

        Parameters
        ----------
        thresholds : array-like, optional
            Thresholds to evaluate. Defaults to 20 log-spaced points.
        ax : matplotlib Axes, optional
            If a list/tuple of two axes is passed, plot xi on ax[0] and
            sigma* on ax[1]. If a single axes, plot xi only.

        Returns
        -------
        StabilityResult
        """
        if thresholds is None:
            thresholds = self._default_thresholds(20)
        thresholds = np.asarray(thresholds, dtype=float)

        xi_vals = np.full(len(thresholds), np.nan)
        xi_se_vals = np.full(len(thresholds), np.nan)
        sigma_star_vals = np.full(len(thresholds), np.nan)
        sigma_star_se_vals = np.full(len(thresholds), np.nan)
        n_above = np.zeros(len(thresholds), dtype=int)

        for i, u in enumerate(thresholds):
            excesses = self.claims[self.claims > u] - u
            n = len(excesses)
            n_above[i] = n
            if n < 5:
                continue
            xi, sigma, xi_se, sigma_se = _fit_gpd_fast(excesses)
            xi_vals[i] = xi
            xi_se_vals[i] = xi_se
            # sigma* = sigma - xi * u; delta method SE
            sigma_star_vals[i] = sigma - xi * u
            # Approximation: SE(sigma*) ≈ SE(sigma) (ignoring correlation)
            sigma_star_se_vals[i] = sigma_se

        if ax is not None:
            if hasattr(ax, "__len__"):
                self._plot_stability(ax[0], thresholds, xi_vals, xi_se_vals, "xi (shape)")
                self._plot_stability(
                    ax[1], thresholds, sigma_star_vals, sigma_star_se_vals, "sigma* (modified scale)"
                )
            else:
                self._plot_stability(ax, thresholds, xi_vals, xi_se_vals, "xi (shape)")

        return StabilityResult(
            thresholds=thresholds,
            xi_values=xi_vals,
            xi_se=xi_se_vals,
            sigma_star_values=sigma_star_vals,
            sigma_star_se=sigma_star_se_vals,
            n_above=n_above,
        )

    def _plot_stability(self, ax, thresholds, values, se, label):
        valid = ~np.isnan(values)
        ax.plot(thresholds[valid], values[valid], "g-o", markersize=3, label=label)
        ax.fill_between(
            thresholds[valid],
            values[valid] - 2 * se[valid],
            values[valid] + 2 * se[valid],
            alpha=0.2,
            color="green",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Threshold u")
        ax.set_ylabel(label)
        ax.set_title(f"Parameter Stability: {label}\n(choose threshold where stable)")
        ax.legend()

    def hill_plot(
        self,
        k_range: Optional[Sequence[int]] = None,
        ax=None,
        bootstrap_ci: bool = True,
        n_boot: int = 500,
    ) -> HillResult:
        """Hill estimator for the tail index.

        The Hill estimator H_k = (1/k) * sum_{i=1}^{k} log(X_{(n-i+1)} / X_{(n-k)})
        estimates 1/xi for Pareto-type tails (xi > 0). It is inconsistent when
        xi <= 0, so the plot should show a stable region with positive values.

        Look for a stable plateau in the plot — this gives a visual estimate
        of xi = 1/H_k.

        Parameters
        ----------
        k_range : sequence of int, optional
            Number of top order statistics to use. Defaults to 5 to n//2.
        ax : matplotlib Axes, optional
            If provided, plot to this axes.
        bootstrap_ci : bool
            Whether to compute bootstrap confidence intervals.
        n_boot : int
            Number of bootstrap replicates.

        Returns
        -------
        HillResult
        """
        n = len(self.claims)
        if k_range is None:
            k_min = max(5, n // 20)
            k_max = n // 2
            k_values = np.arange(k_min, k_max + 1, max(1, (k_max - k_min) // 60))
        else:
            k_values = np.asarray(k_range, dtype=int)

        hill = np.full(len(k_values), np.nan)
        thresholds = np.full(len(k_values), np.nan)
        lower_ci = None
        upper_ci = None

        for i, k in enumerate(k_values):
            if k >= n:
                continue
            top_k = self._sorted[:k]
            cutoff = self._sorted[k]
            thresholds[i] = cutoff
            if cutoff <= 0:
                continue
            ratios = np.log(top_k / cutoff)
            hill[i] = np.mean(ratios)

        if bootstrap_ci and np.any(~np.isnan(hill)):
            rng = np.random.default_rng(42)
            boot_hills = np.full((n_boot, len(k_values)), np.nan)
            for b in range(n_boot):
                sample = rng.choice(self.claims, size=n, replace=True)
                sorted_sample = np.sort(sample)[::-1]
                for i, k in enumerate(k_values):
                    if k >= n:
                        continue
                    cutoff = sorted_sample[k]
                    if cutoff <= 0:
                        continue
                    boot_hills[b, i] = np.mean(np.log(sorted_sample[:k] / cutoff))
            lower_ci = np.nanpercentile(boot_hills, 2.5, axis=0)
            upper_ci = np.nanpercentile(boot_hills, 97.5, axis=0)

        if ax is not None:
            valid = ~np.isnan(hill)
            ax.plot(k_values[valid], hill[valid], "r-", linewidth=1.2, label="Hill H_k")
            if lower_ci is not None:
                ax.fill_between(
                    k_values[valid], lower_ci[valid], upper_ci[valid], alpha=0.2, color="red"
                )
            ax.set_xlabel("k (number of top order statistics)")
            ax.set_ylabel("H_k = 1/xi estimate")
            ax.set_title("Hill Plot\n(stable plateau => Pareto tail; plateau height => 1/xi)")
            ax.legend()

        return HillResult(
            k_values=k_values,
            hill_estimates=hill,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            threshold_values=thresholds,
        )

    def auto_threshold(
        self, method: str = "solari", alpha: float = 0.05
    ) -> float:
        """Automated threshold suggestion.

        This is a starting point for visual inspection, NOT a definitive answer.
        Always examine mrl_plot() and parameter_stability_plot() before trusting
        any automatic threshold.

        Parameters
        ----------
        method : str
            'solari' — select the threshold where mean excess plot linearity
            is maximally consistent using a goodness-of-fit test at level alpha.
            Falls back to 90th percentile if the algorithm is inconclusive.
        alpha : float
            Significance level for the underlying test.

        Returns
        -------
        float
            Suggested threshold. Always annotate with caution in reports.
        """
        if method != "solari":
            raise ValueError(f"Unknown method '{method}'. Only 'solari' supported in v0.1.")

        thresholds = self._default_thresholds(40)
        best_u = np.percentile(self.claims, 90)  # default fallback
        best_pvalue = -1.0

        for u in thresholds:
            excesses = self.claims[self.claims > u] - u
            n = len(excesses)
            if n < 10:
                break
            # Test GPD fit quality using KS test against fitted GPD
            try:
                from scipy.stats import genpareto, kstest
                xi, _, sigma = genpareto.fit(excesses, floc=0)
                stat, pvalue = kstest(excesses, lambda x: genpareto.cdf(x, xi, loc=0, scale=sigma))
                if pvalue > alpha and pvalue > best_pvalue:
                    best_pvalue = pvalue
                    best_u = u
            except Exception:
                continue

        return float(best_u)
