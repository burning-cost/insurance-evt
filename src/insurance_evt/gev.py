"""GEV fitting for block maxima (annual maximum claims).

The Generalised Extreme Value distribution arises from the block maxima approach
— fitting annual (or seasonal) maximum claim amounts. For a reinsurer modelling
peak annual loss, GEV is natural. For individual claim severity, GPD (threshold
exceedance) usually uses data more efficiently.

GEV(xi, mu, sigma):
  - xi > 0: Frechet domain (heavy-tailed; Pareto-type)
  - xi = 0: Gumbel domain (exponential-tailed)
  - xi < 0: Weibull domain (bounded upper tail)

For UK property catastrophe, xi is usually in (0.1, 0.4) — Frechet.

Return level formula:
  x_T = mu - sigma/xi * [1 - (-log(1 - 1/T))^(-xi)]  for xi != 0
  x_T = mu - sigma * log(-log(1 - 1/T))               for xi = 0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import optimize, stats

from .results import GEVFitResult


def _gev_return_level(mu: float, sigma: float, xi: float, T: float) -> float:
    """GEV return level for return period T years."""
    p = 1 - 1 / T
    y = -np.log(p)  # reduced variate
    if abs(xi) < 1e-8:
        return mu - sigma * np.log(y)
    return mu + sigma / xi * (y ** (-xi) - 1)


class GEVFitter:
    """Fit GEV distribution to annual (block) maxima.

    Parameters
    ----------
    block_maxima : array-like
        Annual maximum claim amounts. Typically n=20-50 years of data.
        Must be strictly positive.

    Examples
    --------
    >>> import numpy as np
    >>> maxima = np.array([1.2e6, 2.3e6, 0.8e6, 3.1e6, ...])  # annual maxima
    >>> fitter = GEVFitter(maxima)
    >>> result = fitter.fit()
    >>> fitter.return_level(200)  # 1-in-200 year level
    8.3e6
    >>> fitter.domain_of_attraction()
    'Frechet'
    """

    def __init__(self, block_maxima: np.ndarray) -> None:
        self.block_maxima = np.asarray(block_maxima, dtype=float)
        if np.any(self.block_maxima <= 0):
            raise ValueError("All block maxima must be strictly positive.")
        if len(self.block_maxima) < 5:
            raise ValueError(
                f"Only {len(self.block_maxima)} block maxima. Need at least 5 for GEV fitting."
            )
        self._fit_result: GEVFitResult | None = None

    def fit(self) -> GEVFitResult:
        """Fit GEV to block maxima via MLE.

        Returns
        -------
        GEVFitResult
        """
        data = self.block_maxima
        n = len(data)

        # scipy.stats.genextreme: c = -xi (note sign convention)
        # genextreme(c, loc=mu, scale=sigma) with c = -xi
        c0, mu0, sigma0 = stats.genextreme.fit(data)
        xi0 = -c0

        def neg_loglik(params):
            xi_, mu_, sigma_ = params
            if sigma_ <= 0:
                return 1e12
            c_ = -xi_
            lp = stats.genextreme.logpdf(data, c_, loc=mu_, scale=sigma_)
            if np.any(~np.isfinite(lp)):
                return 1e12
            return -np.sum(lp)

        bounds = [(-2.0, 5.0), (None, None), (1e-6, None)]
        result = optimize.minimize(
            neg_loglik,
            [xi0, mu0, sigma0],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        xi_mle, mu_mle, sigma_mle = result.x
        loglik = -result.fun

        # Standard errors via numerical Hessian
        from scipy.optimize import approx_fprime

        h = 1e-5

        def grad(p):
            return approx_fprime(p, neg_loglik, h)

        hess = np.zeros((3, 3))
        p0 = np.array([xi_mle, mu_mle, sigma_mle])
        for i in range(3):
            ei = np.zeros(3)
            ei[i] = h
            hess[:, i] = (grad(p0 + ei) - grad(p0 - ei)) / (2 * h)

        try:
            cov = np.linalg.inv(hess)
            xi_se = np.sqrt(max(cov[0, 0], 0))
            mu_se = np.sqrt(max(cov[1, 1], 0))
            sigma_se = np.sqrt(max(cov[2, 2], 0))
        except np.linalg.LinAlgError:
            xi_se = abs(xi_mle) / np.sqrt(n)
            mu_se = sigma_mle / np.sqrt(n)
            sigma_se = sigma_mle / np.sqrt(n)

        self._fit_result = GEVFitResult(
            xi=float(xi_mle),
            mu=float(mu_mle),
            sigma=float(sigma_mle),
            xi_se=float(xi_se),
            mu_se=float(mu_se),
            sigma_se=float(sigma_se),
            n_blocks=n,
            loglik=float(loglik),
        )
        return self._fit_result

    def return_level(self, return_period: float) -> float:
        """Compute the GEV return level for a given return period.

        Parameters
        ----------
        return_period : float
            Return period in years (e.g. 200 for 1-in-200).

        Returns
        -------
        float
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before return_level().")
        fit = self._fit_result
        return _gev_return_level(fit.mu, fit.sigma, fit.xi, return_period)

    def profile_likelihood_ci(
        self, return_period: float, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Profile likelihood CI for a GEV return level.

        The CI for x_T is obtained by reparametrising the log-likelihood in terms
        of x_T (holding x_T fixed, optimising over remaining parameters) and
        finding where the profile log-likelihood crosses the chi2 cutoff.

        Parameters
        ----------
        return_period : float
            Return period in years.
        alpha : float
            Significance level (0.05 => 95% CI).

        Returns
        -------
        (lower, upper) : tuple of float
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before profile_likelihood_ci().")

        fit = self._fit_result
        data = self.block_maxima
        loglik_max = fit.loglik
        cutoff = loglik_max - stats.chi2.ppf(1 - alpha, df=1) / 2

        x_T_mle = self.return_level(return_period)

        def profile_loglik(x_T):
            """Maximise loglik over (mu, sigma) for fixed return level x_T."""
            # Express mu in terms of x_T, xi, sigma:
            # x_T = mu + sigma/xi * (y^(-xi) - 1) where y = -log(1 - 1/T)
            # So mu = x_T - sigma/xi * (y^(-xi) - 1)
            y = -np.log(1 - 1 / return_period)

            def neg_ll_2d(params):
                xi_, sigma_ = params
                if sigma_ <= 0:
                    return 1e12
                if abs(xi_) < 1e-8:
                    mu_ = x_T + sigma_ * np.log(y)
                else:
                    mu_ = x_T - sigma_ / xi_ * (y ** (-xi_) - 1)
                c_ = -xi_
                lp = stats.genextreme.logpdf(data, c_, loc=mu_, scale=sigma_)
                if np.any(~np.isfinite(lp)):
                    return 1e12
                return -np.sum(lp)

            r = optimize.minimize(
                neg_ll_2d,
                [fit.xi, fit.sigma],
                method="L-BFGS-B",
                bounds=[(-2.0, 5.0), (1e-6, None)],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            return -r.fun

        def objective(x_T):
            return profile_loglik(x_T) - cutoff

        sigma_xT = fit.xi_se * abs(x_T_mle) + fit.sigma_mle_approx_se(x_T_mle)
        lo_start = x_T_mle - 5 * (fit.sigma + fit.xi_se * abs(x_T_mle))
        hi_start = x_T_mle + 10 * (fit.sigma + fit.xi_se * abs(x_T_mle))
        lo_start = max(lo_start, 0.0)

        try:
            lo = optimize.brentq(objective, lo_start, x_T_mle, xtol=1e-3, maxiter=100)
        except ValueError:
            lo = lo_start
        try:
            hi = optimize.brentq(objective, x_T_mle, hi_start, xtol=1e-3, maxiter=100)
        except ValueError:
            hi = hi_start

        return (float(lo), float(hi))

    def domain_of_attraction(self) -> str:
        """Classify the domain of attraction from the fitted xi.

        Returns
        -------
        str
            'Frechet' (xi > 0.01), 'Gumbel' (|xi| <= 0.01), or 'Weibull' (xi < -0.01).
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() before domain_of_attraction().")
        return self._fit_result.domain_of_attraction


# Monkey-patch to provide SE approximation for return level CI
def _gev_fit_result_sigma_mle_approx_se(self, x_T: float) -> float:
    """Rough SE approximation for return level (used in CI search bounds)."""
    return self.sigma_se * abs(x_T) / max(abs(self.mu), 1.0)


GEVFitResult.sigma_mle_approx_se = _gev_fit_result_sigma_mle_approx_se
