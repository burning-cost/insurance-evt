"""
insurance-evt: Extreme Value Theory for catastrophic insurance claim severity.

This library provides production-grade EVT workflows for UK insurance pricing
teams. It fills the gap left by pyextremes (time series focused, no claim
severity interface) and raw scipy (distributions without the workflow).

The central output is the 1-in-200 year return level with profile likelihood
confidence intervals — the number Solvency II internal model teams need.

Key classes
-----------
ThresholdSelector
    Diagnostic plots for GPD threshold selection: mean residual life,
    parameter stability, Hill estimator. The diagnostics are the product;
    auto_threshold() is a sanity check, not a verdict.

GPDFitter
    MLE for GPD with support for right-censored (open claims) and
    left-truncated (reinsurer attachment) data. Profile likelihood CIs
    for xi and sigma — asymmetric, unlike Wald intervals.

GEVFitter
    GEV fitting to annual block maxima. Natural for subsidence and
    portfolio-level catastrophe modelling.

ReturnLevelCalculator
    GPD return levels with profile likelihood CI band. The T-year return
    level is the main deliverable for capital and reinsurance work.

ExcessLayerCalculator
    XL layer pure premium from GPD. The first Python implementation of the
    actuarial ExcessGPD formula (previously R-only via ReIns package).

SolvencyIIReport
    Structured 1-in-200 year report for internal model documentation.

InsurancePreset (FLOOD, TPBI, SUBSIDENCE, LARGE_FIRE, STORM)
    Peril-specific parameter expectations and threshold guidance.

Quick start
-----------
>>> import numpy as np
>>> from insurance_evt import ThresholdSelector, GPDFitter, ReturnLevelCalculator
>>> from insurance_evt import SolvencyIIReport, ExcessLayerCalculator

>>> # Step 1: choose threshold
>>> sel = ThresholdSelector(claims)
>>> sel.mrl_plot()
>>> u = sel.auto_threshold()   # suggestion only — always examine the plot

>>> # Step 2: fit GPD
>>> fit = GPDFitter(claims, threshold=u).fit()
>>> print(fit)
GPDFitResult(xi=0.3241 ± 0.0812, sigma=187000 ± 23000, n=87, threshold=500000)

>>> # Step 3: return levels
>>> calc = ReturnLevelCalculator(fit, lambda_annual=4.2)
>>> calc.return_level(200)
3_850_000
>>> calc.return_level_ci(200)
(2_100_000, 7_400_000)   # asymmetric — upper tail matters more

>>> # Step 4: Solvency II report
>>> SolvencyIIReport(calc).generate()['interpretation']
'The 1-in-200 year claim severity is estimated at £3.9m (95% CI: £2.1m to £7.4m)...'

>>> # Step 5: reinsurance layer pricing
>>> xl = ExcessLayerCalculator(fit, lambda_annual=4.2)
>>> xl.layer_table([500_000, 1_000_000, 2_000_000], [500_000, 1_000_000, 2_000_000])

Notes
-----
This library handles cross-sectional claim severity data (arrays of amounts).
It does not handle time series; there is no DatetimeIndex interface.
For time series extremes, use pyextremes. For claim severity EVT, use this.

Non-stationarity: standard EVT assumes iid claims. For weather-driven UK perils
(flood, subsidence), climate trends may violate this. Document the assumption.
A non-stationary GEV interface is planned for v0.2.

Right-censored data: open/unsettled claims where ultimate is unknown. Use the
right_censoring parameter in GPDFitter. Ignoring censoring when > 10% of
exceedances are open claims will bias xi upward (Poudyal & Brazauskas 2023).
"""

from .threshold import ThresholdSelector
from .gpd import GPDFitter
from .gev import GEVFitter
from .returns import ReturnLevelCalculator
from .reinsurance import ExcessLayerCalculator
from .solvency import SolvencyIIReport
from .presets import (
    InsurancePreset,
    FLOOD,
    TPBI,
    SUBSIDENCE,
    LARGE_FIRE,
    STORM,
    get_preset,
    PRESETS,
)
from .results import (
    GPDFitResult,
    GEVFitResult,
    MRLResult,
    StabilityResult,
    HillResult,
    ReturnLevelCurve,
)
from . import plots

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "ThresholdSelector",
    "GPDFitter",
    "GEVFitter",
    "ReturnLevelCalculator",
    "ExcessLayerCalculator",
    "SolvencyIIReport",
    # Presets
    "InsurancePreset",
    "FLOOD",
    "TPBI",
    "SUBSIDENCE",
    "LARGE_FIRE",
    "STORM",
    "get_preset",
    "PRESETS",
    # Result types
    "GPDFitResult",
    "GEVFitResult",
    "MRLResult",
    "StabilityResult",
    "HillResult",
    "ReturnLevelCurve",
    # Submodule
    "plots",
    "__version__",
]
