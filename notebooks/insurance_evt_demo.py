# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-evt: Extreme Value Theory for Insurance Claim Severity
# MAGIC
# MAGIC This notebook demonstrates the full workflow for EVT-based catastrophe pricing:
# MAGIC
# MAGIC 1. Generate synthetic UK flood/TPBI claims (realistic severity distribution)
# MAGIC 2. Threshold selection diagnostics (MRL, parameter stability, Hill plot)
# MAGIC 3. GPD fitting with standard and censored MLE
# MAGIC 4. Profile likelihood confidence intervals for xi and return levels
# MAGIC 5. Solvency II 1-in-200 year report
# MAGIC 6. Reinsurance XL layer pricing table
# MAGIC 7. GEV for annual maxima (subsidence example)

# COMMAND ----------

# MAGIC %pip install insurance-evt matplotlib pandas scipy numpy

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Databricks

from scipy.stats import genpareto, lognorm

np.random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Claims Data
# MAGIC
# MAGIC We generate 5,000 UK flood claims over 30 years with:
# MAGIC - Lognormal body (most claims under £200k)
# MAGIC - GPD tail above £300k (threshold for EVT analysis)
# MAGIC - True parameters: xi=0.32 (Frechet, heavy tail), sigma=180k
# MAGIC - ~4.5 exceedances above £300k per year on average
# MAGIC
# MAGIC This matches typical UK domestic flood: heavy tail from total losses in
# MAGIC high-value properties, claims concentrated in 2000, 2007, 2013-14, 2020 events.

# COMMAND ----------

# True parameters for synthetic data
THRESHOLD = 300_000
XI_TRUE = 0.32
SIGMA_TRUE = 180_000
N_YEARS = 30
LAMBDA_TRUE = 4.5  # exceedances per year above 300k

# Generate body claims (lognormal)
n_body = 4000
body_claims = lognorm.rvs(s=1.3, scale=np.exp(10.2), size=n_body)
body_claims = body_claims[body_claims > 0]

# Generate tail claims (GPD exceedances)
n_tail = int(LAMBDA_TRUE * N_YEARS)  # ~135 exceedances
tail_excesses = genpareto.rvs(XI_TRUE, loc=0, scale=SIGMA_TRUE, size=n_tail)
tail_claims = THRESHOLD + tail_excesses

# Combine
all_claims = np.concatenate([body_claims, tail_claims])
print(f"Total claims: {len(all_claims):,}")
print(f"Claims above threshold: {np.sum(all_claims > THRESHOLD):,}")
print(f"Max claim: £{all_claims.max():,.0f}")
print(f"True xi={XI_TRUE}, sigma=£{SIGMA_TRUE:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Threshold Selection Diagnostics
# MAGIC
# MAGIC The threshold choice is the most consequential decision in EVT.
# MAGIC We examine three diagnostics before committing.

# COMMAND ----------

from insurance_evt import ThresholdSelector
from insurance_evt import plots

sel = ThresholdSelector(all_claims)

# Mean Residual Life plot
thresholds = np.percentile(all_claims, np.linspace(60, 95, 35))
mrl_result = sel.mrl_plot(thresholds=thresholds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plots.plot_mrl(mrl_result, ax=axes[0], title="Mean Residual Life Plot\n(Look for linear region)")
axes[0].axvline(THRESHOLD, color='red', linestyle='--', label=f'True threshold £{THRESHOLD:,}')
axes[0].legend()

# Parameter stability
thresholds_stab = np.percentile(all_claims, np.linspace(65, 93, 20))
stability = sel.parameter_stability_plot(thresholds=thresholds_stab, ax=[axes[1], axes[1]])

# Replot stability properly
axes[1].clear()
stab_result = sel.parameter_stability_plot(thresholds=thresholds_stab)
plots.plot_parameter_stability(stab_result, axes=[axes[1], axes[1]])
axes[1].axvline(THRESHOLD, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# Hill plot (tail index diagnostic)
fig, ax = plt.subplots(figsize=(9, 4))
hill = sel.hill_plot(bootstrap_ci=True, n_boot=300, ax=ax)
ax.axhline(1/XI_TRUE, color='red', linestyle='--', linewidth=1.5, label=f'True 1/xi = {1/XI_TRUE:.2f}')
ax.legend()
ax.set_title(f"Hill Plot — plateau should be near {1/XI_TRUE:.2f}\n(= 1/xi where xi = {XI_TRUE})")
plt.tight_layout()
display(fig)
plt.close()

# Summary
hill_df = hill.to_dataframe()
print(f"\nHill estimates range: [{hill_df['hill_estimate'].min():.3f}, {hill_df['hill_estimate'].max():.3f}]")
print(f"Median Hill H_k = {hill_df['hill_estimate'].median():.3f} (implies xi ≈ {1/hill_df['hill_estimate'].median():.3f})")

# COMMAND ----------

# Automated threshold suggestion
suggested_u = sel.auto_threshold()
print(f"auto_threshold() suggests: £{suggested_u:,.0f}")
print(f"True threshold: £{THRESHOLD:,}")
print(f"\nNOTE: This is a starting point for visual inspection, not a definitive answer.")
print(f"The MRL and parameter stability plots above are the real diagnostics.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. GPD Fitting — Standard MLE

# COMMAND ----------

from insurance_evt import GPDFitter

fitter = GPDFitter(all_claims, threshold=THRESHOLD)
fit = fitter.fit()

print("=== GPD Fit Results ===")
print(fit)
print(f"\nxi (shape):     {fit.xi:.4f} ± {fit.xi_se:.4f}  (true: {XI_TRUE})")
print(f"sigma (scale):  £{fit.sigma:,.0f} ± £{fit.sigma_se:,.0f}  (true: £{SIGMA_TRUE:,})")
print(f"n exceedances:  {fit.n_exceedances}")
print(f"log-likelihood: {fit.loglik:.2f}")
print(f"\nTail index 1/xi: {fit.tail_index:.3f}")

# COMMAND ----------

# Profile likelihood CIs (asymmetric)
print("=== Profile Likelihood CIs ===")
lo_xi, hi_xi = fitter.profile_likelihood_ci('xi', alpha=0.05)
lo_sigma, hi_sigma = fitter.profile_likelihood_ci('sigma', alpha=0.05)

print(f"\nxi:    {fit.xi:.4f}  (95% PL CI: {lo_xi:.4f}, {hi_xi:.4f})")
print(f"       Wald:    (± {1.96*fit.xi_se:.4f})")
print(f"       PL upper extent: {hi_xi - fit.xi:.4f}")
print(f"       PL lower extent: {fit.xi - lo_xi:.4f}")
print(f"       Upper asymmetry: {(hi_xi - fit.xi)/(fit.xi - lo_xi):.2f}x")

print(f"\nsigma: £{fit.sigma:,.0f}  (95% PL CI: £{lo_sigma:,.0f}, £{hi_sigma:,.0f})")
print(f"\nTrue xi={XI_TRUE} in CI: {lo_xi <= XI_TRUE <= hi_xi}")

# COMMAND ----------

# GPD diagnostic plots (PP-plot and QQ-plot)
fig = plots.plot_gpd_diagnostic(
    fit,
    fitter._exceedances,
    title=f"GPD Fit Diagnostics (xi={fit.xi:.3f}, n={fit.n_exceedances})"
)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Return Level Calculation

# COMMAND ----------

from insurance_evt import ReturnLevelCalculator

lambda_annual = fit.n_exceedances / N_YEARS
calc = ReturnLevelCalculator(fit, lambda_annual=lambda_annual)
calc.attach_data(fitter._exceedances)

print(f"Lambda (exceedances/year): {lambda_annual:.2f}")
print(f"\nReturn levels:")
for T in [20, 50, 100, 200, 500, 1000]:
    rl = calc.return_level(T)
    print(f"  1-in-{T:4d} year:  £{rl:>10,.0f}")

# COMMAND ----------

# Return level curve with profile likelihood CI
print("Computing profile likelihood CI curve (this takes ~30-60 seconds)...")
T_arr = [20, 50, 100, 200, 500]

# Use Wald for speed in demo; use 'profile' for final reports
curve_wald = calc.return_level_curve(T_arr, method='wald')

fig = plots.plot_return_level_curve(
    curve_wald,
    title=f"Return Level Curve — UK Flood Claims\n(xi={fit.xi:.3f}, lambda={lambda_annual:.1f}/yr)",
    log_scale=True
)
display(fig)
plt.close()

# Display as table
print("\nReturn level table (Wald CI):")
df_curve = curve_wald.to_dataframe()
df_curve.columns = ['Return Period (yr)', 'Return Level (£)', 'Lower 95% CI (£)', 'Upper 95% CI (£)']
for col in df_curve.columns[1:]:
    df_curve[col] = df_curve[col].map(lambda x: f"£{x:,.0f}")
display(df_curve)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Solvency II 1-in-200 Year Report

# COMMAND ----------

from insurance_evt import SolvencyIIReport

report_obj = SolvencyIIReport(calc)
result = report_obj.generate(alpha_ci=0.05)

print("=" * 60)
print("SOLVENCY II INTERNAL MODEL — CATASTROPHE SCR")
print("Peril: UK Domestic Flood | Method: GPD (Peaks-over-Threshold)")
print("=" * 60)
print()
print(result['interpretation'])
print()
print("Model parameters:")
print(f"  Threshold:           £{result['threshold']:,.0f}")
print(f"  xi (shape):          {result['xi']:.4f} ± {result['xi_se']:.4f}")
print(f"  sigma (scale):       £{result['sigma']:,.0f}")
print(f"  Lambda (annual):     {result['lambda_annual']:.2f}")
print(f"  n exceedances:       {result['n_exceedances']}")
print(f"  n censored:          {result['n_censored']}")
print()
print("Actuarial warnings:")
for i, w in enumerate(result['warnings'], 1):
    print(f"  {i}. {w}")

# DataFrame version
print("\nDataFrame output (for model logging):")
display(report_obj.to_dataframe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Reinsurance XL Layer Pricing

# COMMAND ----------

from insurance_evt import ExcessLayerCalculator

xl = ExcessLayerCalculator(fit, lambda_annual=lambda_annual)

# Mean excess at various retentions
print("Mean excess (expected loss above retention per occurrence):")
retentions = [350_000, 500_000, 750_000, 1_000_000, 1_500_000]
for M in retentions:
    me = xl.mean_excess(M)
    print(f"  E[X-{M/1e3:.0f}k | X>{M/1e3:.0f}k] = £{me:,.0f}")

# COMMAND ----------

# Full layer pricing table
retentions_table = [350_000, 500_000, 500_000, 750_000, 1_000_000, 2_000_000]
limits_table =     [350_000, 500_000, None,    750_000, 1_000_000, 1_000_000]

df_layers = xl.layer_table(retentions_table, limits_table)
df_layers['layer_desc'] = [
    f"£{int(r/1e3)}k xs £{int(r/1e3)}k" if l < 1e15 else f"Unlimited xs £{int(r/1e3)}k"
    for r, l in zip(df_layers['retention'], df_layers['limit'])
]
df_layers['pure_premium_formatted'] = df_layers['pure_premium'].map(lambda x: f"£{x:,.0f}")
df_layers['rate_on_line_formatted'] = df_layers['rate_on_line'].map(
    lambda x: f"{x:.1%}" if np.isfinite(x) else "N/A"
)

print("XL Layer Pure Premium Table:")
display(df_layers[['layer_desc', 'retention', 'mean_excess', 'pure_premium_formatted', 'rate_on_line_formatted']])

# Bar chart
fig = plots.plot_layer_table(
    df_layers[df_layers['limit'] < 1e15],
    title=f"XL Layer Pure Premiums — UK Flood\n(xi={fit.xi:.3f}, lambda={lambda_annual:.1f}/yr)"
)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Censored MLE — TPBI with Open Claims
# MAGIC
# MAGIC Large TPBI claims (catastrophic injuries) settle over 5-15 years.
# MAGIC At any given extract date, many large claims are still open.
# MAGIC Ignoring this censoring overstates xi, leading to over-reserving or
# MAGIC over-priced reinsurance.

# COMMAND ----------

# Generate TPBI claims with right-censoring
np.random.seed(77)

TPBI_THRESHOLD = 500_000
XI_TPBI = 0.40
SIGMA_TPBI = 250_000
N_TPBI_YEARS = 20

n_tpbi_body = 2000
tpbi_body = lognorm.rvs(s=1.5, scale=np.exp(11.0), size=n_tpbi_body)
tpbi_body = tpbi_body[tpbi_body < TPBI_THRESHOLD]

n_tpbi_tail = 60  # 3 exceedances/year
tpbi_excesses = genpareto.rvs(XI_TPBI, loc=0, scale=SIGMA_TPBI, size=n_tpbi_tail)
tpbi_tail = TPBI_THRESHOLD + tpbi_excesses

# 20% of tail claims are open (right-censored at £1.5m policy limit indicator)
n_censored = int(0.20 * n_tpbi_tail)
censor_flag_tail = np.zeros(n_tpbi_tail, dtype=bool)
# Censor the largest n_censored claims (large claims more likely to be open)
sort_idx = np.argsort(tpbi_tail)
censor_flag_tail[sort_idx[-n_censored:]] = True

tpbi_all = np.concatenate([tpbi_body, tpbi_tail])
censor_all = np.concatenate([np.zeros(len(tpbi_body), dtype=bool), censor_flag_tail])

print(f"TPBI dataset: {len(tpbi_all):,} claims, {n_tpbi_tail} above £{TPBI_THRESHOLD/1e6:.1f}m")
print(f"Censored claims (open/unsettled): {censor_flag_tail.sum()} ({100*censor_flag_tail.mean():.0f}% of tail)")
print(f"True parameters: xi={XI_TPBI}, sigma=£{SIGMA_TPBI:,}")

# COMMAND ----------

# Fit with and without censoring
fitter_standard = GPDFitter(tpbi_all, threshold=TPBI_THRESHOLD)
fit_standard = fitter_standard.fit()

fitter_censored = GPDFitter(tpbi_all, threshold=TPBI_THRESHOLD, right_censoring=censor_all)
fit_censored = fitter_censored.fit()

print("=== Impact of Ignoring Censoring ===")
print(f"{'Parameter':<15} {'Standard MLE':<20} {'Censored MLE':<20} {'Truth':<10}")
print("-" * 65)
print(f"{'xi':<15} {fit_standard.xi:<20.4f} {fit_censored.xi:<20.4f} {XI_TPBI:<10}")
print(f"{'sigma':<15} £{fit_standard.sigma:<18,.0f} £{fit_censored.sigma:<18,.0f} £{SIGMA_TPBI:<8,}")
print(f"{'n_censored':<15} {0:<20} {fit_censored.n_censored:<20}")
print()

# Return level comparison
for f_name, f in [("Standard MLE", fit_standard), ("Censored MLE", fit_censored)]:
    lam = f.n_exceedances / N_TPBI_YEARS
    c = ReturnLevelCalculator(f, lambda_annual=lam)
    rl = c.return_level(200)
    print(f"1-in-200 ({f_name}): £{rl:,.0f}")

print(f"\nOver-statement from ignoring censoring: £{(ReturnLevelCalculator(fit_standard, fit_standard.n_exceedances/N_TPBI_YEARS).return_level(200) - ReturnLevelCalculator(fit_censored, fit_censored.n_exceedances/N_TPBI_YEARS).return_level(200)):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. GEV for Annual Maxima — Subsidence Example
# MAGIC
# MAGIC For subsidence, annual maximum portfolio loss is often more natural than
# MAGIC individual claims. Claims cluster by event year (1976, 1995, 2003, 2018, 2022)
# MAGIC and independence of individual claims cannot be assumed.

# COMMAND ----------

from insurance_evt import GEVFitter

np.random.seed(55)

# Synthetic annual max subsidence loss (20 years)
from scipy.stats import genextreme

XI_SUBS = 0.15
MU_SUBS = 2_500_000
SIGMA_SUBS = 800_000
n_years_subs = 35

c_subs = -XI_SUBS  # scipy sign convention
annual_maxima = genextreme.rvs(c_subs, loc=MU_SUBS, scale=SIGMA_SUBS, size=n_years_subs)
annual_maxima = np.abs(annual_maxima)

print(f"Annual max subsidence loss ({n_years_subs} years):")
print(f"  Min: £{annual_maxima.min():,.0f}  Max: £{annual_maxima.max():,.0f}")
print(f"  True: xi={XI_SUBS}, mu=£{MU_SUBS:,}, sigma=£{SIGMA_SUBS:,}")

# COMMAND ----------

gev_fitter = GEVFitter(annual_maxima)
gev_fit = gev_fitter.fit()

print("=== GEV Fit Results ===")
print(gev_fit)
print(f"\nxi:     {gev_fit.xi:.4f} ± {gev_fit.xi_se:.4f}  (true: {XI_SUBS})")
print(f"mu:     £{gev_fit.mu:,.0f} ± £{gev_fit.mu_se:,.0f}  (true: £{MU_SUBS:,})")
print(f"sigma:  £{gev_fit.sigma:,.0f} ± £{gev_fit.sigma_se:,.0f}  (true: £{SIGMA_SUBS:,})")
print(f"\nDomain of attraction: {gev_fitter.domain_of_attraction()}")

print("\nGEV return levels:")
for T in [20, 50, 100, 200, 500]:
    rl = gev_fitter.return_level(T)
    print(f"  1-in-{T:4d} year:  £{rl:>10,.0f}")

rl_200 = gev_fitter.return_level(200)
lo_gev, hi_gev = gev_fitter.profile_likelihood_ci(200)
print(f"\n1-in-200 year (GEV): £{rl_200:,.0f} (95% CI: £{lo_gev:,.0f} to £{hi_gev:,.0f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Preset Comparison
# MAGIC
# MAGIC The library includes typical xi ranges for UK perils.
# MAGIC Use these as sanity checks on your fitted parameters.

# COMMAND ----------

from insurance_evt import FLOOD, TPBI, SUBSIDENCE, LARGE_FIRE, STORM

presets = [FLOOD, TPBI, SUBSIDENCE, LARGE_FIRE, STORM]
rows = []
for p in presets:
    rows.append({
        'Peril': p.name.upper(),
        'Preferred Method': p.preferred_method.upper(),
        'Threshold Percentile': f"{p.threshold_percentile:.0f}th",
        'Typical xi Range': f"{p.typical_xi_range[0]:.2f}–{p.typical_xi_range[1]:.2f}",
    })

df_presets = pd.DataFrame(rows)
print("Insurance Peril Presets:")
display(df_presets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC
# MAGIC 1. **Threshold selection**: MRL + parameter stability + Hill plot — visual diagnostics, not automated prescription
# MAGIC 2. **GPD MLE**: standard, right-censored (open claims), and left-truncated (reinsurer data)
# MAGIC 3. **Profile likelihood CI**: asymmetric intervals, correctly capturing positive skew in xi uncertainty
# MAGIC 4. **Return levels**: 1-in-200 year estimate with CI band
# MAGIC 5. **Solvency II report**: structured output with interpretation and actuarial warnings
# MAGIC 6. **XL layer pricing**: ExcessGPD formula — annual expected cost of each layer
# MAGIC 7. **GEV**: annual maxima approach for clustered perils (subsidence)
# MAGIC
# MAGIC The library is at PyPI: `pip install insurance-evt`
# MAGIC GitHub: https://github.com/burning-cost/insurance-evt
