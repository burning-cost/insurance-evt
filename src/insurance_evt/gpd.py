"""GPD fitting for excess claims over a threshold.

The Generalised Pareto Distribution is the asymptotically justified model for
exceedances over a high threshold (Pickands-Balkema-de Haan theorem). This
module handles three common data complications in insurance:

- Standard MLE on clean exceedances
- Right-censored MLE: open claims where the ultimate is unknown. The censored
  observations contribute log(1 - F(x)) instead of log f(x).
- Left-truncated MLE: reinsurer data where only claims above a deductible are
  observed. Likelihood must be conditioned on X > deductible.

Profile likelihood CIs are asymmetric (unlike Wald CIs from Fisher information).
For the tail shape parameter xi, the asymmetry can be substantial — especially
at small samples. Always use profile likelihood, not xi_se.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy import optimize, stats

from .results import GPDFitResult


def _gpd_logpdf(excesses: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """Log-PDF of GPD(xi, sigma) evaluated at excess values."""
    if sigma <= 0:
        return np.full(len(excesses), -np.inf)
    y = excesses / sigma
    if abs(xi) < 1e-8:
        # Exponential limit
        return -np.log(sigma) - y
    arg = 1 + xi * y
    if np.any(arg <= 0):
        return np.full(len(excesses), -np.inf)
    return -np.log(sigma) - (1 + 1 / xi) * np.log(arg)


def _gpd_logsf(excess: float, xi: float, sigma: float) -> float:
    """Log-survival function of GPD at a single excess value."""
    if sigma <= 0:
        return -np.inf
    y = excess / sigma
    if abs(xi) < 1e-8:
        return -y
    arg = 1 + xi * y
    if arg <= 0:
        return 0.0  # below support
    return (-1 / xi) * np.log(arg)


def _gpd_cdf(excesses: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """CDF of GPD(xi, sigma)."""
    y = excesses / sigma
    if abs(xi) < 1e-8:
        return 1 - np.exp(-y)
    arg = 1 + xi * y
    arg = np.maximum(arg, 0.0)
    return 1 - arg ** (-1 / xi)


def _neg_loglik_standard(
    params: Tuple[float, float], excesses: np.ndarray
) -> float:
    """Standard GPD negative log-likelihood."""
    xi, sigma = params
    if sigma <= 0:
        return 1e12
    lp = _gpd_logpdf(excesses, xi, sigma)
    if np.any(~np.isfinite(lp)):
        return 1e12
    return -np.sum(lp)


def _neg_loglik_censored(
    params: Tuple[float, float],
    excesses: np.ndarray,
    censored: np.ndarray,
) -> float:
    """GPD negative log-likelihood with right censoring.

    censored[i] = True means the i-th observation is a lower bound (open claim).
    These contribute log(1 - F(excess_i)) = log SF(excess_i) to the likelihood.
    """
    xi, sigma = params
    if sigma <= 0:
        return 1e12
    observed = ~censored
    nll = 0.0
    if np.any(observed):
        lp = _gpd_logpdf(excesses[observed], xi, sigma)
        if np.any(~np.isfinite(lp)):
            return 1e12
        nll -= np.sum(lp)
    if np.any(censored):
        for exc in excesses[censored]:
            ls = _gpd_logsf(exc, xi, sigma)
            nll -= ls
    return nll


def _neg_loglik_truncated(
    params: Tuple[float, float],
    excesses: np.ndarray,
    trunc_above_threshold: float,
) -> float:
    """GPD negative log-likelihood with left truncation.

    When a reinsurer only sees claims above attachment point M > threshold u,
    the observed excesses are y = x - u, but only those with y > trunc_above_threshold
    (i.e. M - u) are in the data. The conditional likelihood is:

        L = prod_i f(y_i) / S(trunc_above_threshold)
    """
    xi, sigma = params
    if sigma <= 0:
        return 1e12
    lp = _gpd_logpdf(excesses, xi, sigma)
    if np.any(~np.isfinite(lp)):
        return 1e12
    ls = _gpd_logsf(trunc_above_threshold, xi, sigma)
    n = len(excesses)
    return -(np.sum(lp) - n * ls)


def _fisher_information_se(
    params: Tuple[float, float],
    neg_loglik_fn,
    **kwargs,
) -> Tuple[float, float]:
    """Numerical standard errors from observed Fisher information."""
    from scipy.optimize import approx_fprime

    h = 1e-5

    def grad(p):
        return approx_fprime(p, lambda x: neg_loglik_fn(x, **kwargs), h)

    hess = np.zeros((2, 2))
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = h
        hess[:, i] = (grad(np.array(params) + ei) - grad(np.array(params) - ei)) / (2 * h)
    xi_mle, sigma_mle = params
    n = len(kwargs.get("excesses", [1]))
    try:
        cov = np.linalg.inv(hess)
        xi_se = np.sqrt(max(cov[0, 0], 0))
        sigma_se = np.sqrt(max(cov[1, 1], 0))
        # Guard against numerical zero: minimum plausible SE
        if xi_se < 1e-10:
            xi_se = max(abs(xi_mle) * 0.05, 0.01) / np.sqrt(max(n, 1))
        if sigma_se < 1e-10:
            sigma_se = sigma_mle * 0.05 / np.sqrt(max(n, 1))
    except np.linalg.LinAlgError:
        xi_se = max(abs(xi_mle), 0.01) / np.sqrt(max(n, 1))
        sigma_se = sigma_mle / np.sqrt(max(n, 1))
    return xi_se, sigma_se


class GPDFitter:
    """Fit a Generalised Pareto Distribution to claim exceedances.

    Parameters
    ----------
    claims : array-like
        Raw claim amounts (all of them, not pre-filtered). The fitter selects
        exceedances above `threshold` internally.
    threshold : float
        The threshold u. Only claims > u are used.
    left_truncation : float, optional
        If the data only includes claims above this attachment point (reinsurer
        perspective), set left_truncation = attachment_point. Must be >= threshold.
    right_censoring : array-like of bool, optional
        Boolean array same length as `claims`. True = this claim is open/unsettled
        and the reported amount is a lower bound on the ultimate.

    Examples
    --------
    >>> fitter = GPDFitter(claims, threshold=500_000)
    >>> result = fitter.fit()
    >>> result.xi, result.sigma
    (0.32, 187000)

    >>> lower, upper = fitter.profile_likelihood_ci('xi', alpha=0.05)
    >>> print(f"xi in ({lower:.3f}, {upper:.3f})")
    """

    def __init__(
        self,
        claims: np.ndarray,
        threshold: float,
        left_truncation: Optional[float] = None,
        right_censoring: Optional[np.ndarray] = None,
    ) -> None:
        self.claims = np.asarray(claims, dtype=float)
        self.threshold = float(threshold)
        self.left_truncation = left_truncation

        # Select exceedances
        mask = self.claims > self.threshold
        self._exceedances = self.claims[mask] - self.threshold

        if right_censoring is not None:
            rc = np.asarray(right_censoring, dtype=bool)
            if len(rc) != len(self.claims):
                raise ValueError(
                    f"right_censoring length ({len(rc)}) must match claims length ({len(self.claims)})"
                )
            self._censored = rc[mask]
        else:
            self._censored = np.zeros(len(self._exceedances), dtype=bool)

        if left_truncation is not None:
            if left_truncation < threshold:
                raise ValueError(
                    f"left_truncation ({left_truncation}) must be >= threshold ({threshold})"
                )
            self._trunc_above_threshold = float(left_truncation - threshold)
            # Only keep exceedances above the truncation point
            trunc_mask = self._exceedances > self._trunc_above_threshold
            self._exceedances = self._exceedances[trunc_mask]
            self._censored = self._censored[trunc_mask]
        else:
            self._trunc_above_threshold = None

        self._fit_result: Optional[GPDFitResult] = None

    @property
    def n_exceedances(self) -> int:
        return len(self._exceedances)

    @property
    def n_censored(self) -> int:
        return int(np.sum(self._censored))

    def fit(self, method: Literal["mle"] = "mle") -> GPDFitResult:
        """Fit GPD to exceedances.

        Parameters
        ----------
        method : str
            Currently only 'mle'. MCMC planned for v0.2.

        Returns
        -------
        GPDFitResult
        """
        if method != "mle":
            raise ValueError(f"method='{method}' not supported. Use 'mle'.")

        n = self.n_exceedances
        if n < 5:
            raise ValueError(
                f"Only {n} exceedances above threshold {self.threshold}. "
                "Need at least 5 for reliable fitting. Try a lower threshold."
            )

        excesses = self._exceedances
        censored = self._censored

        # Initial parameter estimate (method of moments)
        m1 = np.mean(excesses)
        m2 = np.mean(excesses**2)
        ratio = m2 / (2 * m1**2)
        xi0 = 1 - 1 / max(ratio, 1.01) if ratio > 1 else 0.2
        xi0 = np.clip(xi0, -0.5, 2.0)
        sigma0 = m1 * (1 - xi0) if abs(1 - xi0) > 0.01 else m1

        if self._trunc_above_threshold is not None:
            neg_ll = lambda p: _neg_loglik_truncated(
                p, excesses, self._trunc_above_threshold
            )
        elif np.any(censored):
            neg_ll = lambda p: _neg_loglik_censored(p, excesses, censored)
        else:
            neg_ll = lambda p: _neg_loglik_standard(p, excesses)

        # Optimise with bounds: sigma > 0, xi in (-1, 10)
        bounds = [(-1.0, 10.0), (1e-6, None)]
        result = optimize.minimize(
            neg_ll,
            [xi0, sigma0],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            # Retry with Nelder-Mead (no bounds, but xi > -1 enforced in neg_ll)
            result2 = optimize.minimize(
                neg_ll,
                [xi0, sigma0],
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
            )
            if result2.fun < result.fun:
                result = result2

        xi_mle, sigma_mle = result.x
        loglik = -result.fun

        # Standard errors
        if self._trunc_above_threshold is not None:
            xi_se, sigma_se = _fisher_information_se(
                (xi_mle, sigma_mle), _neg_loglik_truncated,
                excesses=excesses, trunc_above_threshold=self._trunc_above_threshold
            )
        elif np.any(censored):
            xi_se, sigma_se = _fisher_information_se(
                (xi_mle, sigma_mle), _neg_loglik_censored,
                excesses=excesses, censored=censored
            )
        else:
            xi_se, sigma_se = _fisher_information_se(
                (xi_mle, sigma_mle), _neg_loglik_standard,
                excesses=excesses
            )

        self._fit_result = GPDFitResult(
            xi=float(xi_mle),
            sigma=float(sigma_mle),
            xi_se=float(xi_se),
            sigma_se=float(sigma_se),
            n_exceedances=n,
            loglik=float(loglik),
            threshold=self.threshold,
            n_censored=self.n_censored,
            left_truncation=self.left_truncation,
            method=method,
        )
        return self._fit_result

    def profile_likelihood_ci(
        self, param: Literal["xi", "sigma"] = "xi", alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Profile likelihood confidence interval for xi or sigma.

        Unlike Wald intervals (xi ± z * xi_se), profile likelihood intervals
        are asymmetric and correctly capture the positive skew in xi uncertainty.
        For small samples (n < 50), the difference is substantial.

        The 100(1-alpha)% CI is the set of parameter values where the profile
        log-likelihood exceeds L_max - chi2(1, 1-alpha) / 2.
        For 95% CI: cutoff = L_max - 1.92.

        Parameters
        ----------
        param : 'xi' or 'sigma'
            Which parameter to construct CI for.
        alpha : float
            Confidence level (default 0.05 => 95% CI).

        Returns
        -------
        (lower, upper) : tuple of float
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before profile_likelihood_ci().")
        if param not in ("xi", "sigma"):
            raise ValueError(f"param must be 'xi' or 'sigma', got '{param}'")

        fit = self._fit_result
        excesses = self._exceedances
        censored = self._censored

        if self._trunc_above_threshold is not None:
            neg_ll_fn = lambda p: _neg_loglik_truncated(
                p, excesses, self._trunc_above_threshold
            )
        elif np.any(censored):
            neg_ll_fn = lambda p: _neg_loglik_censored(p, excesses, censored)
        else:
            neg_ll_fn = lambda p: _neg_loglik_standard(p, excesses)

        loglik_max = fit.loglik
        cutoff = loglik_max - stats.chi2.ppf(1 - alpha, df=1) / 2

        if param == "xi":
            fixed_idx = 0
            free_idx = 1
            mle_val = fit.xi
            mle_se = fit.xi_se
            bounds_free = [(1e-6, None)]
        else:  # sigma
            fixed_idx = 1
            free_idx = 0
            mle_val = fit.sigma
            mle_se = fit.sigma_se
            bounds_free = [(-1.0, 10.0)]

        def profile_loglik(val):
            """Maximise loglik over free param for fixed param=val."""
            if param == "xi":
                free_init = [fit.sigma]
            else:
                free_init = [fit.xi]

            def neg_ll_1d(free_params):
                if param == "xi":
                    p = [val, free_params[0]]
                else:
                    p = [free_params[0], val]
                return neg_ll_fn(p)

            r = optimize.minimize(
                neg_ll_1d,
                free_init,
                method="L-BFGS-B",
                bounds=bounds_free,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            return -r.fun

        def objective(val):
            return profile_loglik(val) - cutoff

        # Search range: ±5 SE from MLE, expanding if needed
        lo_start = mle_val - 5 * max(mle_se, abs(mle_val) * 0.1 + 1e-6)
        hi_start = mle_val + 5 * max(mle_se, abs(mle_val) * 0.1 + 1e-6)

        if param == "xi":
            lo_start = max(lo_start, -0.99)

        # Find lower bound
        try:
            lo = optimize.brentq(objective, lo_start, mle_val, xtol=1e-6, maxiter=100)
        except ValueError:
            lo = lo_start

        # Find upper bound
        try:
            hi = optimize.brentq(objective, mle_val, hi_start, xtol=1e-6, maxiter=100)
        except ValueError:
            hi = hi_start

        return (float(lo), float(hi))
