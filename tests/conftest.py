"""Shared test fixtures for insurance-evt tests."""

import numpy as np
import pytest
from scipy.stats import genpareto, genextreme


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def pareto_claims(rng):
    """Synthetic Pareto-type claims: lognormal body + GPD tail.

    GPD parameters: xi=0.3, sigma=200_000, threshold=500_000
    Annual rate: ~4 exceedances per year over 500k threshold.
    """
    # Body: lognormal claims under 500k
    n_body = 2000
    body = rng.lognormal(mean=10.5, sigma=1.2, size=n_body)
    body = body[body < 500_000]

    # Tail: GPD exceedances above 500k
    n_tail = 120
    threshold = 500_000
    xi_true, sigma_true = 0.30, 200_000
    excesses = genpareto.rvs(xi_true, loc=0, scale=sigma_true, size=n_tail, random_state=rng)
    tail = threshold + excesses

    claims = np.concatenate([body, tail])
    return claims, threshold, xi_true, sigma_true


@pytest.fixture(scope="module")
def gpd_excesses(rng):
    """Clean GPD excesses for fitting tests: xi=0.3, sigma=150_000."""
    xi, sigma = 0.30, 150_000
    return genpareto.rvs(xi, loc=0, scale=sigma, size=200, random_state=rng), xi, sigma


@pytest.fixture(scope="module")
def small_excesses(rng):
    """Small sample GPD excesses: n=25, xi=0.25."""
    xi, sigma = 0.25, 100_000
    return genpareto.rvs(xi, loc=0, scale=sigma, size=25, random_state=rng), xi, sigma


@pytest.fixture(scope="module")
def gev_maxima(rng):
    """Annual maxima from GEV(xi=0.2, mu=1e6, sigma=300_000)."""
    xi_true, mu_true, sigma_true = 0.2, 1_000_000, 300_000
    c = -xi_true  # scipy convention
    maxima = genextreme.rvs(c, loc=mu_true, scale=sigma_true, size=50, random_state=rng)
    maxima = np.abs(maxima)  # ensure positive
    return maxima, xi_true, mu_true, sigma_true


@pytest.fixture(scope="module")
def censored_claims(rng):
    """Claims with ~15% right censoring: open/unsettled large claims."""
    xi, sigma, threshold = 0.35, 180_000, 400_000
    n = 100
    excesses = genpareto.rvs(xi, loc=0, scale=sigma, size=n, random_state=rng)
    claims = threshold + excesses

    # Censor 15 claims above the 85th percentile (open large claims)
    censor_threshold = np.percentile(claims, 85)
    censored = claims > censor_threshold
    # Censored claims are reported at censor_threshold (known lower bound)
    claims_observed = np.where(censored, censor_threshold, claims)
    all_claims = np.concatenate([
        rng.lognormal(11, 1.0, size=500),  # body claims
        claims_observed
    ])
    all_censored = np.concatenate([
        np.zeros(500, dtype=bool),
        censored
    ])
    return all_claims, all_censored, threshold, xi, sigma
