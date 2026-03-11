"""Result dataclasses for insurance-evt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GPDFitResult:
    """Result from GPDFitter.fit().

    Attributes
    ----------
    xi : float
        Shape parameter. xi > 0 => heavy tail (Pareto-type).
        xi = 0 => exponential tail. xi < 0 => bounded tail.
    sigma : float
        Scale parameter. Must be positive.
    xi_se : float
        Standard error of xi from observed Fisher information.
    sigma_se : float
        Standard error of sigma from observed Fisher information.
    n_exceedances : int
        Number of claims above the threshold.
    loglik : float
        Log-likelihood at the MLE.
    threshold : float
        The threshold u used for fitting.
    n_censored : int
        Number of right-censored observations among exceedances.
    left_truncation : float or None
        Left truncation point if used, else None.
    method : str
        Fitting method ('mle').
    """

    xi: float
    sigma: float
    xi_se: float
    sigma_se: float
    n_exceedances: int
    loglik: float
    threshold: float
    n_censored: int = 0
    left_truncation: Optional[float] = None
    method: str = "mle"

    @property
    def tail_index(self) -> Optional[float]:
        """1/xi — the Pareto tail index. None if xi <= 0."""
        if self.xi > 0:
            return 1.0 / self.xi
        return None

    def __repr__(self) -> str:
        return (
            f"GPDFitResult(xi={self.xi:.4f} ± {self.xi_se:.4f}, "
            f"sigma={self.sigma:.0f} ± {self.sigma_se:.0f}, "
            f"n={self.n_exceedances}, threshold={self.threshold:.0f})"
        )


@dataclass
class GEVFitResult:
    """Result from GEVFitter.fit().

    Attributes
    ----------
    xi : float
        Shape parameter (also written as gamma or eta).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    xi_se, mu_se, sigma_se : float
        Standard errors.
    n_blocks : int
        Number of block maxima used.
    loglik : float
        Log-likelihood at MLE.
    """

    xi: float
    mu: float
    sigma: float
    xi_se: float
    mu_se: float
    sigma_se: float
    n_blocks: int
    loglik: float

    @property
    def domain_of_attraction(self) -> str:
        """Classify tail behaviour."""
        if self.xi > 0.01:
            return "Frechet"
        elif self.xi < -0.01:
            return "Weibull"
        else:
            return "Gumbel"

    def __repr__(self) -> str:
        return (
            f"GEVFitResult(xi={self.xi:.4f} ± {self.xi_se:.4f}, "
            f"mu={self.mu:.0f}, sigma={self.sigma:.0f}, "
            f"n_blocks={self.n_blocks})"
        )


@dataclass
class MRLResult:
    """Result from ThresholdSelector.mrl_plot()."""

    thresholds: np.ndarray
    mean_excess: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    n_above: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "threshold": self.thresholds,
                "mean_excess": self.mean_excess,
                "lower_ci": self.lower_ci,
                "upper_ci": self.upper_ci,
                "n_above": self.n_above,
            }
        )


@dataclass
class StabilityResult:
    """Result from ThresholdSelector.parameter_stability_plot()."""

    thresholds: np.ndarray
    xi_values: np.ndarray
    xi_se: np.ndarray
    sigma_star_values: np.ndarray  # modified scale: sigma* = sigma - xi * threshold
    sigma_star_se: np.ndarray
    n_above: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "threshold": self.thresholds,
                "xi": self.xi_values,
                "xi_se": self.xi_se,
                "sigma_star": self.sigma_star_values,
                "sigma_star_se": self.sigma_star_se,
                "n_above": self.n_above,
            }
        )


@dataclass
class HillResult:
    """Result from ThresholdSelector.hill_plot()."""

    k_values: np.ndarray
    hill_estimates: np.ndarray
    lower_ci: Optional[np.ndarray]
    upper_ci: Optional[np.ndarray]
    threshold_values: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "k": self.k_values,
                "hill_estimate": self.hill_estimates,
                "threshold": self.threshold_values,
            }
        )
        if self.lower_ci is not None:
            df["lower_ci"] = self.lower_ci
            df["upper_ci"] = self.upper_ci
        return df


@dataclass
class ReturnLevelCurve:
    """Return level curve with confidence intervals."""

    return_periods: np.ndarray
    return_levels: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    alpha: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "return_period": self.return_periods,
                "return_level": self.return_levels,
                "lower_ci": self.lower_ci,
                "upper_ci": self.upper_ci,
            }
        )
