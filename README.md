# insurance-evt

Extreme Value Theory for catastrophic insurance claim severity. GPD and GEV fitting with profile likelihood confidence intervals, censored MLE, excess layer pricing, and Solvency II reporting.

## The problem

UK insurers pricing flood, TPBI, subsidence, and large fire need to estimate the 1-in-200 year claim — the Solvency II SCR benchmark. Standard tools do not help:

- **scipy**: provides `genpareto` and `genextreme`, but just the distributions. No threshold selection diagnostics, no profile likelihood CIs, no reinsurance layer pricing.
- **pyextremes**: excellent for time series extremes (river gauges, wind speed), but requires a `DatetimeIndex`. Insurance claim data is cross-sectional — there is no datetime. It also has no excess layer pricing.
- **R's ReIns package**: has `ExcessGPD()`, profile likelihood, truncation/censoring support. But it's R.

This library is the Python gap. It wraps scipy's GPD/GEV with the workflow that insurance pricing teams actually need.

## What's different

**Profile likelihood CIs** — not Wald intervals. For the shape parameter xi, Wald CIs (xi ± z * se) are symmetric and wrong. Profile likelihood CIs are asymmetric and correctly capture the positive skew in return level uncertainty. At n=50 exceedances, the difference between upper bounds is typically 30-50%.

**Censored MLE** — open/unsettled TPBI claims are right-censored: the ultimate is unknown, but the reported amount is a lower bound. Standard MLE on censored data overstates xi by ~15% (Poudyal & Brazauskas 2023). The `right_censoring` parameter handles this.

**Left-truncated MLE** — reinsurers only see claims above their attachment point. The conditional likelihood `f(x | x > attachment)` is different from the unconditional. Set `left_truncation` and the likelihood is adjusted automatically.

**ExcessGPD formula** — the annual expected cost of an XL layer, closed-form from GPD parameters. This is the output reinsurance pricing teams need. Previously available only in R's ReIns package. This is the first Python implementation.

## Install

```bash
pip install insurance-evt
```

## Quick start

```python
import numpy as np
from insurance_evt import ThresholdSelector, GPDFitter, ReturnLevelCalculator
from insurance_evt import SolvencyIIReport, ExcessLayerCalculator

# Step 1: choose a threshold
sel = ThresholdSelector(claims)
mrl = sel.mrl_plot()    # look for linearity
stability = sel.parameter_stability_plot()  # look for stability
u = sel.auto_threshold()  # algorithmic suggestion — examine the plot

# Step 2: fit GPD
fitter = GPDFitter(claims, threshold=u)
fit = fitter.fit()
print(fit)
# GPDFitResult(xi=0.3241 ± 0.0812, sigma=187000 ± 23000, n=87, threshold=500000)

# Step 3: profile likelihood CI for xi (asymmetric)
lo, hi = fitter.profile_likelihood_ci('xi')
print(f"xi: {fit.xi:.3f} ({lo:.3f}, {hi:.3f})")
# xi: 0.324 (0.183, 0.521)  <-- upper wider than lower for heavy tail

# Step 4: return levels
n_years = 30  # years of data
calc = ReturnLevelCalculator(fit, lambda_annual=fit.n_exceedances / n_years)
calc.attach_data(fitter._exceedances)

print(f"1-in-200 year: £{calc.return_level(200):,.0f}")
lo, hi = calc.return_level_ci(200, method='profile')
print(f"95% CI: £{lo:,.0f} to £{hi:,.0f}")

# Step 5: Solvency II report
report = SolvencyIIReport(calc).generate()
print(report['interpretation'])
# "The 1-in-200 year claim severity is estimated at £4.2m (95% CI: £2.8m to £7.6m)..."

# Step 6: XL layer pricing
xl = ExcessLayerCalculator(fit, lambda_annual=fit.n_exceedances / n_years)
df = xl.layer_table(
    retentions=[500_000, 750_000, 1_000_000, 2_000_000],
    limits   =[500_000, 500_000, 1_000_000, 1_000_000],
)
print(df[['retention', 'limit', 'pure_premium', 'rate_on_line']])
```

## With censored claims

```python
# open_flag: True for open/unsettled claims (lower bound on ultimate)
fitter = GPDFitter(claims, threshold=500_000, right_censoring=open_flag)
fit = fitter.fit()
# Censored MLE: open claims contribute log(1-F(x)) instead of log f(x)
```

## With reinsurer data (left truncation)

```python
# Reinsurer only sees claims > 750k; threshold set at 500k for model
fitter = GPDFitter(
    claims,
    threshold=500_000,
    left_truncation=750_000
)
fit = fitter.fit()
# Conditional likelihood: f(x | x > 750k) correct for data selection
```

## Peril presets

```python
from insurance_evt import FLOOD, TPBI, SUBSIDENCE, LARGE_FIRE, get_preset

print(TPBI.typical_xi_range)   # (0.3, 0.55)
print(TPBI.threshold_percentile)  # 95.0
print(TPBI.notes[:200])  # right-censoring guidance...

# Use as sanity check on fitted xi
fit = GPDFitter(claims, threshold=u).fit()
lo, hi = TPBI.typical_xi_range
if not (lo <= fit.xi <= hi):
    print(f"Warning: xi={fit.xi:.3f} outside typical TPBI range {TPBI.typical_xi_range}")
```

## GEV for annual maxima (subsidence, portfolio-level)

```python
from insurance_evt import GEVFitter

fitter = GEVFitter(annual_maxima)  # annual maximum claim per year
fit = fitter.fit()
print(fitter.domain_of_attraction())  # 'Frechet' for heavy-tailed
print(f"1-in-200: £{fitter.return_level(200):,.0f}")
lo, hi = fitter.profile_likelihood_ci(200)
```

## Diagnostic plots

```python
import matplotlib.pyplot as plt
from insurance_evt import plots

sel = ThresholdSelector(claims)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mrl_result = sel.mrl_plot()
plots.plot_mrl(mrl_result, ax=axes[0])
stability = sel.parameter_stability_plot()
plots.plot_parameter_stability(stability, axes=[axes[1]])

# Hill plot
hill = sel.hill_plot(bootstrap_ci=True, n_boot=200)
plots.plot_hill(hill)

# Return level curve
curve = calc.return_level_curve([10, 20, 50, 100, 200, 500], method='profile')
plots.plot_return_level_curve(curve)

# GPD PP/QQ plots
plots.plot_gpd_diagnostic(fit, fitter._exceedances)
```

## A note on threshold selection

There is no automated threshold selection method that reliably outperforms expert visual inspection. `auto_threshold()` uses a KS-test-based selection as a starting point. It is a starting point — not a verdict.

The mean residual life plot and parameter stability plot are the diagnostic tools. Two analysts looking at the same plots may pick thresholds that differ by 10-15%. That's normal, and the profile likelihood CI on the return level captures most of the resulting uncertainty.

## A note on non-stationarity

Standard EVT assumes independent and identically distributed claims. For UK flood and subsidence, this assumption is increasingly questionable: the frequency of extreme rainfall events and hot/dry summers is changing. This library does not model non-stationarity (planned for v0.2). All reports include a stationarity warning. Document this assumption when submitting to your Internal Model Approval Process.

## Background

The asymptotic foundations are in Coles (2001), *An Introduction to Statistical Modeling of Extreme Values* (Springer). The censored/truncated MLE follows Poudyal & Brazauskas (2023) and Albrecher & Beirlant (2024, arXiv:2511.22272). The ExcessGPD formula is from Reynkens et al. (2017), *Insurance: Mathematics and Economics* 77, 65–77.

## License

MIT. Built by [Burning Cost](https://github.com/burning-cost).
