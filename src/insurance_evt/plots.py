"""Diagnostic and summary plots for insurance EVT analysis.

All plot functions return a matplotlib Figure object and accept an optional
`ax` (or `axes`) parameter for embedding in larger figures. The functions
do not call plt.show() — callers control display.

Design note: plotting is separated from computation. The result dataclasses
(MRLResult, StabilityResult, ReturnLevelCurve) carry all computed values;
these functions just render them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .results import (
    GPDFitResult,
    GEVFitResult,
    HillResult,
    MRLResult,
    ReturnLevelCurve,
    StabilityResult,
)


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_mrl(result: MRLResult, ax=None, title: Optional[str] = None):
    """Mean residual life plot.

    Parameters
    ----------
    result : MRLResult
        From ThresholdSelector.mrl_plot().
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    valid = ~np.isnan(result.mean_excess)
    ax.plot(
        result.thresholds[valid],
        result.mean_excess[valid],
        "b-o",
        markersize=4,
        linewidth=1.5,
        label="Mean excess",
    )
    ax.fill_between(
        result.thresholds[valid],
        result.lower_ci[valid],
        result.upper_ci[valid],
        alpha=0.2,
        color="steelblue",
        label="95% CI",
    )
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("E[X − u | X > u]")
    ax.set_title(title or "Mean Residual Life Plot")
    ax.legend(fontsize=9)

    # Secondary axis: n_above
    ax2 = ax.twinx()
    ax2.plot(result.thresholds, result.n_above, "gray", linestyle=":", alpha=0.6)
    ax2.set_ylabel("n above threshold", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    return fig


def plot_parameter_stability(result: StabilityResult, axes=None, title: Optional[str] = None):
    """Parameter stability plot (xi and sigma*).

    Parameters
    ----------
    result : StabilityResult
        From ThresholdSelector.parameter_stability_plot().
    axes : array of 2 matplotlib Axes, optional
        If None, create a new figure with two subplots.
    title : str, optional

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_mpl()
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig = axes[0].get_figure()

    ax1, ax2 = axes

    valid = ~np.isnan(result.xi_values)

    ax1.plot(
        result.thresholds[valid],
        result.xi_values[valid],
        "g-o",
        markersize=4,
        label="xi (shape)",
    )
    ax1.fill_between(
        result.thresholds[valid],
        result.xi_values[valid] - 2 * result.xi_se[valid],
        result.xi_values[valid] + 2 * result.xi_se[valid],
        alpha=0.2,
        color="green",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Threshold u")
    ax1.set_ylabel("xi")
    ax1.set_title("Shape Parameter Stability")
    ax1.legend(fontsize=9)

    valid2 = ~np.isnan(result.sigma_star_values)
    ax2.plot(
        result.thresholds[valid2],
        result.sigma_star_values[valid2],
        "purple",
        marker="o",
        markersize=4,
        label="sigma* (modified scale)",
    )
    ax2.fill_between(
        result.thresholds[valid2],
        result.sigma_star_values[valid2] - 2 * result.sigma_star_se[valid2],
        result.sigma_star_values[valid2] + 2 * result.sigma_star_se[valid2],
        alpha=0.2,
        color="purple",
    )
    ax2.set_xlabel("Threshold u")
    ax2.set_ylabel("sigma* = sigma - xi * u")
    ax2.set_title("Modified Scale Stability")
    ax2.legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_hill(result: HillResult, ax=None, title: Optional[str] = None):
    """Hill estimator plot.

    Parameters
    ----------
    result : HillResult
        From ThresholdSelector.hill_plot().
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    valid = ~np.isnan(result.hill_estimates)
    ax.plot(
        result.k_values[valid],
        result.hill_estimates[valid],
        "r-",
        linewidth=1.5,
        label="Hill H_k",
    )
    if result.lower_ci is not None:
        ax.fill_between(
            result.k_values[valid],
            result.lower_ci[valid],
            result.upper_ci[valid],
            alpha=0.2,
            color="red",
            label="95% bootstrap CI",
        )
    ax.set_xlabel("k (top order statistics)")
    ax.set_ylabel("H_k = 1/xi estimate")
    ax.set_title(title or "Hill Plot\n(plateau height = 1/xi; flat region = stable)")
    ax.legend(fontsize=9)
    return fig


def plot_return_level_curve(
    curve: ReturnLevelCurve,
    ax=None,
    title: Optional[str] = None,
    log_scale: bool = True,
):
    """Return level curve with confidence interval band.

    Parameters
    ----------
    curve : ReturnLevelCurve
        From ReturnLevelCalculator.return_level_curve().
    ax : matplotlib Axes, optional
    title : str, optional
    log_scale : bool
        If True, use log scale on x-axis (return period).

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    ax.plot(
        curve.return_periods,
        curve.return_levels,
        "b-",
        linewidth=2,
        label="Return level",
    )
    ax.fill_between(
        curve.return_periods,
        curve.lower_ci,
        curve.upper_ci,
        alpha=0.25,
        color="steelblue",
        label=f"{int((1 - curve.alpha) * 100)}% CI",
    )
    ax.axvline(200, color="red", linestyle="--", linewidth=0.8, label="1-in-200 (SII SCR)")

    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Return level (claim amount)")
    ax.set_title(title or "GPD Return Level Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_gpd_diagnostic(
    fit: GPDFitResult,
    excesses: np.ndarray,
    axes=None,
    title: Optional[str] = None,
):
    """GPD fit diagnostic: PP-plot and QQ-plot.

    Parameters
    ----------
    fit : GPDFitResult
    excesses : array-like
        Exceedances above threshold (already subtracted threshold).
    axes : array of 2 matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_mpl()
    from scipy.stats import genpareto

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig = axes[0].get_figure()

    excesses = np.sort(np.asarray(excesses, dtype=float))
    n = len(excesses)
    xi, sigma = fit.xi, fit.sigma

    # Empirical CDF
    emp_probs = np.arange(1, n + 1) / (n + 1)

    # Model CDF
    model_probs = genpareto.cdf(excesses, xi, loc=0, scale=sigma)

    ax1 = axes[0]
    ax1.scatter(model_probs, emp_probs, s=15, alpha=0.7, color="steelblue")
    ax1.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax1.set_xlabel("Model probability")
    ax1.set_ylabel("Empirical probability")
    ax1.set_title("PP-Plot")

    # QQ-plot
    model_quantiles = genpareto.ppf(emp_probs, xi, loc=0, scale=sigma)
    ax2 = axes[1]
    ax2.scatter(model_quantiles, excesses, s=15, alpha=0.7, color="steelblue")
    lims = [
        min(model_quantiles.min(), excesses.min()),
        max(model_quantiles.max(), excesses.max()),
    ]
    ax2.plot(lims, lims, "r--", linewidth=1)
    ax2.set_xlabel("Model quantile")
    ax2.set_ylabel("Empirical quantile")
    ax2.set_title("QQ-Plot")

    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle(
            f"GPD Fit Diagnostics: xi={xi:.3f}, sigma={sigma:.0f}, n={n}",
            fontsize=11,
        )
    fig.tight_layout()
    return fig


def plot_layer_table(layer_df, ax=None, title: Optional[str] = None):
    """Bar chart of layer pure premiums.

    Parameters
    ----------
    layer_df : pd.DataFrame
        From ExcessLayerCalculator.layer_table().
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    retentions = layer_df["retention"].values
    pp = layer_df["pure_premium"].values

    x = np.arange(len(retentions))
    ax.bar(x, pp, color="steelblue", alpha=0.8)
    labels = [f"£{r/1e6:.1f}m" if r >= 1e6 else f"£{r/1e3:.0f}k" for r in retentions]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Pure premium (annual expected cost)")
    ax.set_title(title or "XL Layer Pure Premiums")
    fig.tight_layout()
    return fig
