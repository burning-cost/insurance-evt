"""Insurance peril presets with typical EVT parameter ranges.

These are not magic defaults — they're informed starting points for UK personal
lines perils based on empirical literature and market practice. Use them to:
  1. Sanity-check your MLE output (if xi >> typical_xi_max, revisit data)
  2. Set threshold_percentile as an initial search range
  3. Flag to reviewers which peril regime your analysis is in

All typical_xi_range values are from KB entry 1372 (UK peril parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class InsurancePreset:
    """Configuration preset for a specific insurance peril.

    Attributes
    ----------
    name : str
        Short identifier.
    description : str
        Plain-English description including data source guidance.
    threshold_percentile : float
        Suggested starting percentile for threshold selection (0-100).
        This is a starting point for mrl_plot(), not a prescription.
    typical_xi_range : tuple of float
        (min, max) for xi based on UK market experience.
        If fitted xi falls outside this, re-examine data and threshold.
    preferred_method : str
        'gpd' for threshold exceedance, 'gev' for block maxima.
    notes : str
        Practitioner notes on data quality, truncation, etc.
    """

    name: str
    description: str
    threshold_percentile: float
    typical_xi_range: Tuple[float, float]
    preferred_method: Literal["gpd", "gev"] = "gpd"
    notes: str = ""


FLOOD = InsurancePreset(
    name="flood",
    description=(
        "UK property flood claims — buildings and contents damage from river, "
        "surface water, and coastal flooding events. Data: individual settled "
        "claim amounts from flood years (2000, 2007, 2013-14, 2020)."
    ),
    threshold_percentile=90.0,
    typical_xi_range=(0.10, 0.40),
    preferred_method="gpd",
    notes=(
        "Flood Reinsurance data starts from 2016 for high-risk properties. "
        "Pre/post Flood Re is a structural break — fit GPD separately. "
        "Claims are heavy-tailed due to subsidence interaction in high-water events. "
        "Censoring is unusual for flood (claims settle relatively quickly)."
    ),
)

TPBI = InsurancePreset(
    name="tpbi",
    description=(
        "Third party bodily injury claims above £500k — catastrophic injuries "
        "from motor accidents, Ogden rate sensitive. Data: large loss register, "
        "claims in excess of reference level."
    ),
    threshold_percentile=95.0,
    typical_xi_range=(0.30, 0.55),
    preferred_method="gpd",
    notes=(
        "Right-censoring is significant: large TPBI claims settle over 5-15 years. "
        "Use right_censoring=True for open claims. "
        "Ogden rate changes (2017, 2019) create non-stationarity in effective severity. "
        "Some reinsurers apply left_truncation at their attachment point. "
        "xi typically higher than property perils — unbounded injury awards."
    ),
)

SUBSIDENCE = InsurancePreset(
    name="subsidence",
    description=(
        "UK domestic subsidence claims — ground movement from clay shrinkage. "
        "Claims are highly correlated with hot/dry summers. "
        "Block maxima (annual max per portfolio) are natural for GEV approach."
    ),
    threshold_percentile=90.0,
    typical_xi_range=(0.05, 0.35),
    preferred_method="gev",
    notes=(
        "Subsidence claims cluster by event year (1976, 1995, 2003, 2018, 2022). "
        "Annual maxima approach (GEV) often preferred over POT for this reason. "
        "Strong climate trend: frequency of hot/dry summers increasing. "
        "Standard EVT iid assumption is increasingly questionable — document this. "
        "Typical repair cost £15k-£150k; severity lighter-tailed than TPBI."
    ),
)

LARGE_FIRE = InsurancePreset(
    name="large_fire",
    description=(
        "Commercial large fire claims above £100k — factory, warehouse, "
        "office building total losses. Data: large loss bordereaux."
    ),
    threshold_percentile=95.0,
    typical_xi_range=(0.20, 0.50),
    preferred_method="gpd",
    notes=(
        "Sum insured acts as a policy limit — creates right truncation at "
        "policy limit. For Lloyd's syndicates, claims may be proportional "
        "to sum insured, creating systematic right truncation. "
        "Frequency has been declining (better sprinkler coverage); "
        "this affects lambda_annual — re-estimate from recent years only. "
        "Post-2015 data preferable; older fire claims have different construction mix."
    ),
)

STORM = InsurancePreset(
    name="storm",
    description=(
        "UK domestic storm/wind claims — roof damage, fallen trees, guttering. "
        "Typically lower severity but high frequency; large losses rare. "
        "Named storm events (Ciaran, Babet, Isha 2023-24) drive tail."
    ),
    threshold_percentile=90.0,
    typical_xi_range=(0.05, 0.30),
    preferred_method="gpd",
    notes=(
        "Storm claims are strongly event-driven; independence assumption "
        "requires care — multiple claims per household in same storm. "
        "For EVT at policy level, claims are approximately independent. "
        "For portfolio-level modelling, aggregate storm loss is the right unit "
        "and GEV on annual aggregate may be more appropriate."
    ),
)


# Named access
PRESETS = {
    "flood": FLOOD,
    "tpbi": TPBI,
    "subsidence": SUBSIDENCE,
    "large_fire": LARGE_FIRE,
    "storm": STORM,
}


def get_preset(name: str) -> InsurancePreset:
    """Get a preset by name.

    Parameters
    ----------
    name : str
        One of: 'flood', 'tpbi', 'subsidence', 'large_fire', 'storm'.

    Returns
    -------
    InsurancePreset
    """
    key = name.lower().replace("-", "_")
    if key not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[key]
