# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:01:07 2020
Updated on Wed Aug  6 2026

@author: andre
QUESTION 2 / FORECASTING PILOT:
Implement the proposed model using the data and extract the predicted resolution
time for 3 interesting cases.

NOTE: the rows in forecasting_pilot.csv are issues that were still OPEN at the time
of the snapshot, so they have no resolution date and no ground truth for the
target. This script is therefore a FORECASTING PILOT (a qualitative sanity
check on realistic, still-open cases), not a quantitative model evaluation.
Quantitative evaluation is done on a random 80/20 train/test split inside
6_multi_lin_reg.py, 7_regression_trees.py and 8_survival_analysis.py.

All five model families are applied to the same 324 open issues:
  - regression/tree: predicted TOTAL resolution time from creation (days)
  - Cox / random survival forest: predicted REMAINING days (conditional on how
    long the issue has already been open) plus P(resolve in 30/90/180 days).
    For a like-for-like comparison on the same plot, the survival "total" is
    elapsed_days + median_remaining_days.

The fitted models are loaded from disk: scripts 6/7/8 saved them (and the
out-of-sample pilot survival models) into their directories, so this script is
fully standalone.

Outputs (question2/):
  - predictionComparison.png / predictionComparison_log.png  all five models on
    the 324 open issues (day scale and log scale)
  - forecasting_pilot_predictions.csv   unified per-issue table
  - forecasting_pilot_results.txt       interesting cases + conclusions
"""
import os
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from avro_common import print_section, prepare_forecasting_pilot_features

print_section('PREDICTING (FORECASTING PILOT)')

# Models fitted by the earlier modeling scripts (6_multi_lin_reg.py,
# 7_regression_trees.py, 8_survival_analysis.py) and saved to disk. Running
# this script standalone only needs those files, not the whole pipeline.
linreg = load('multi_lin_reg/linreg.joblib')
tree = load('regression_tree/tree.joblib')
rfr = load('regression_tree/rfr.joblib')
cox_p = load('survival_analysis/cox_pilot.joblib')   # fit on resolved-only (out-of-sample)
rsf_p = load('survival_analysis/rsf_pilot.joblib')

forecasting_pilot = pd.read_csv('data_sets/forecasting_pilot.csv')
print('\nForecasting pilot on issues that were still open at snapshot time:')
print('  shape: {} rows x {} columns'.format(*forecasting_pilot.shape))
print('  statuses: {}'.format(sorted(forecasting_pilot['status'].unique())))

training = pd.read_csv('data_sets/encoded.csv')
training = training.drop('duration',axis=1)

# create directory
if not os.path.exists('question2'):
    os.makedirs('question2')

# Build the SAME 32-feature encoding used by the training pipeline (avro_common
# prepare_forecasting_pilot_features), so every model - including the survival
# ones fitted on that exact encoding - sees features consistent with training.
# NB: The legacy hand-rolled pipeline in this script used test-set reporter counts,
# which disagreed with the survival training encoding.
test_full = prepare_forecasting_pilot_features(forecasting_pilot, training.columns)
test_linreg = test_full.reindex(columns=linreg.feature_names_in_, fill_value=0)

#%% Prediction - regression / trees (log-minutes scale, as trained)
lr_pred = linreg.predict(test_linreg)    # multivariate linear model
tree_pred = tree.predict(test_full)      # regression tree model
rf_pred = rfr.predict(test_full)         # random forest regression model

#%% Prediction - survival models (Cox / random survival forest)
# Elapsed time since creation (days) for each still-open issue.
created = pd.to_datetime(forecasting_pilot['created'], errors='coerce', utc=True).dt.tz_localize(None)
updated = pd.to_datetime(forecasting_pilot['updated'], errors='coerce', utc=True).dt.tz_localize(None)
elapsed = (updated - created).dt.total_seconds().to_numpy() / 86400.0


def conditional_summary(survival_curves, elapsed, horizons=(30, 90, 180)):
    """Given S(t) per subject, return median remaining days and P(resolve in h)."""
    out = np.zeros((len(elapsed), 1 + len(horizons)))
    for i, (ts, ps) in enumerate(survival_curves):
        s0 = float(np.interp(elapsed[i], ts, ps, left=1.0))
        s0 = max(s0, 1e-9)
        # conditional median: first time S(t) <= S(t0) / 2
        target = 0.5 * s0
        hits = np.where(ps <= target)[0]
        med_t = ts[hits[0]] if len(hits) else ts[-1]
        out[i, 0] = max(med_t - elapsed[i], 0.0)
        for j, h in enumerate(horizons):
            sh = float(np.interp(elapsed[i] + h, ts, ps, left=1.0))
            out[i, 1 + j] = 1.0 - sh / s0
    return out


# Cox survival curves
sf_cox = cox_p.predict_survival_function(test_full)
cox_times = sf_cox.index.to_numpy()
cox_curves = [(cox_times, sf_cox[col].to_numpy()) for col in sf_cox.columns]
cox_pilot = conditional_summary(cox_curves, elapsed)

# RSF survival curves (StepFunction objects)
sfs = rsf_p.predict_survival_function(test_full.to_numpy())
rsf_curves = [(np.asarray(sf.x), np.asarray(sf.y)) for sf in sfs]
rsf_pilot = conditional_summary(rsf_curves, elapsed)

#%% Unified forecasting-pilot table (all five models, original day scale)
# Regression/tree models predict the TOTAL days from creation; the survival
# "total" is elapsed + median remaining. Both are directly comparable.
pilot = pd.DataFrame({
    'key': forecasting_pilot['key'],
    'status': forecasting_pilot['status'],
    'elapsed_days': np.round(elapsed, 1),
    'LinearModel_total_days': np.round(np.exp(lr_pred) / 1440, 1),
    'RegrTree_total_days': np.round(np.exp(tree_pred) / 1440, 1),
    'RandomForest_total_days': np.round(np.exp(rf_pred) / 1440, 1),
    'Cox_total_days': np.round(elapsed + cox_pilot[:, 0], 1),
    'Cox_median_remaining': np.round(cox_pilot[:, 0], 1),
    'Cox_P_30d': np.round(cox_pilot[:, 1], 3),
    'Cox_P_90d': np.round(cox_pilot[:, 2], 3),
    'Cox_P_180d': np.round(cox_pilot[:, 3], 3),
    'RSF_total_days': np.round(elapsed + rsf_pilot[:, 0], 1),
    'RSF_median_remaining': np.round(rsf_pilot[:, 0], 1),
    'RSF_P_30d': np.round(rsf_pilot[:, 1], 3),
    'RSF_P_90d': np.round(rsf_pilot[:, 2], 3),
    'RSF_P_180d': np.round(rsf_pilot[:, 3], 3),
})
pilot.to_csv('question2/forecasting_pilot_predictions.csv', index=False)
print('\nSaved unified forecasting-pilot predictions -> question2/forecasting_pilot_predictions.csv')

#%% Prediction comparison plots - ALL five models (survival as total days)
comp = pd.DataFrame({
    'LinearModel': np.exp(lr_pred) / 1440,
    'RegrTree': np.exp(tree_pred) / 1440,
    'RandomForest': np.exp(rf_pred) / 1440,
    'Cox PH': elapsed + cox_pilot[:, 0],
    'Random survival forest': elapsed + rsf_pilot[:, 0],
})
comp.plot(figsize=(12,5),marker='.')
plt.xlabel("Forecasting pilot observations",fontsize=15)
plt.ylabel("Duration (days)",fontsize=15)
plt.title("Forecasting Pilot Prediction - all five models (day scale)",fontsize=16)
plt.grid(True)
plt.savefig('question2/predictionComparison.png')
plt.close()

np.log1p(comp).plot(figsize=(12,5),marker='.')
plt.xlabel("Forecasting pilot observations",fontsize=15)
plt.ylabel("LOG(Duration)",fontsize=15)
plt.title("Forecasting Pilot Prediction - all five models (log scale)",fontsize=16)
plt.grid(True)
plt.savefig('question2/predictionComparison_log.png')
plt.close()

#%% Write results + conclusions to a text file
summary = pd.DataFrame({
    'median_remaining_days': [np.median(cox_pilot[:, 0]), np.median(rsf_pilot[:, 0])],
    'P(resolve in 30d)': [cox_pilot[:, 1].mean(), rsf_pilot[:, 1].mean()],
    'P(resolve in 90d)': [cox_pilot[:, 2].mean(), rsf_pilot[:, 2].mean()],
    'P(resolve in 180d)': [cox_pilot[:, 3].mean(), rsf_pilot[:, 3].mean()],
}, index=['Cox PH', 'Random survival forest']).round(3)

interesting_idx = [321, 120, 235, 231, 77, 243, 113]
interesting = pilot.loc[interesting_idx, [
    'key', 'status', 'elapsed_days', 'LinearModel_total_days', 'RegrTree_total_days',
    'RandomForest_total_days', 'Cox_median_remaining', 'Cox_total_days',
    'RSF_median_remaining', 'RSF_total_days',
]]

conclusions = [
    ('321 (AVRO-1124)', 'Open / New Feature / Major',
     'Regression and tree models predict a quick resolution (~2-3 weeks from '
     'creation), but this issue has already been Open for ~569 days and is very '
     'popular (18 votes, 50 comments, 46 watchers). The survival models agree '
     'with the intuition of the discussion: they expect roughly 77-116 more days '
     '(total ~646-685 days). The short regression/tree totals look unrealistic here; '
     'the survival view better matches the long, active history of the issue.'),
    ('120 (AVRO-939)', 'Patch Available / New Feature / Major',
     'Unassigned ("Patch Available" for ~280 days). Trees and linear model '
     'predict ~12-22 days total, i.e. that it is basically already resolved. '
     'The survival models are far more pessimistic: 231-354 more days '
     '(total ~511-634 days). The status has not changed in almost a year, so the '
     'survival forecast - long remaining time despite the patch - is the cautionary '
     'reading.'),
    ('235 (AVRO-283)', 'Open / Improvement / Major',
     'Stuck in Open for ~1275 days (>3.5 years) with a medium popular thread. '
     'The regression/tree models return absurdly short totals (2.5-3.8 days): '
     'those models have no notion of how long the issue has already been open. '
     'The survival models return 0 remaining days - the conditional survival '
     'curve has already dropped below 50%, i.e. statistically the issue is '
     'unlikely to ever be resolved. This is the clearest illustration of why the '
     'survival conditioning on elapsed time matters.'),
    ('231 (AVRO-341)', 'Open / Improvement / Major',
     'Unassigned for ~322 days. Regression/trees predict ~3-5 days total, again '
     'ignoring the elapsed time. Survival predicts 105-256 more days (total '
     '~427-578 days). The unassigned, ~1-year-old status supports the longer '
     'survival horizon.'),
    ('77 (AVRO-1113)', 'Open / Bug / Minor',
     'A brand-new, unassigned Minor bug (elapsed ~1 day, no votes/comments). '
     'All five models agree on a short horizon (~4-11 days total). New, quiet, '
     'low-priority issues are expected to be fixed quickly or quietly triaged; '
     'the models are consistent here.'),
    ('243 (AVRO-266)', 'Open / Improvement / Major',
     'Another long-stuck improvement (~720 days elapsed, 1 comment). Regression/'
     'trees predict ~5-7 days total, which is not credible given the age. The '
     'survival models give ~54-56 more days (total ~773-776 days), a much more '
     'realistic horizon for an issue that has effectively stalled.'),
    ('113 (AVRO-1455)', 'Patch Available / Bug / Major',
     'A brand-new Bug (~1 day old) by frequent contributor tomwhite, already with '
     'a Patch Available. All models predict a fast resolution (linear 2.8 days, '
     'survival 5-14 remaining days). Here regression/tree and survival agree: '
     'new issue + active contributor + patch ready = quick fix.'),
]

lines = []
lines.append('QUESTION 2 - FORECASTING PILOT: PREDICTED RESOLUTION TIMES')
lines.append('=' * 66)
lines.append('\nNOTE: these are the 324 issues still OPEN at snapshot time (no ground truth).')
lines.append('This is a qualitative sanity check, not a model evaluation (see 6/7/8 scripts).')
lines.append('Regression/tree "total" = predicted resolution time from creation.')
lines.append('Survival "total" = elapsed + median remaining days (conditional on age).\n')

lines.append('---- Interesting cases ----')
lines.append(interesting.to_string(index=True))
lines.append('\n---- Average survival forecast for the 324 open issues ----')
lines.append(summary.to_string())

print()
for model, arr in [('Cox PH', cox_pilot), ('RSF', rsf_pilot)]:
    is_patch = (pilot['status'] == 'Patch Available').to_numpy()
    row = {}
    for label, mask in [('Patch Available', is_patch), ('Other open', ~is_patch)]:
        row['{} n'.format(label)] = int(mask.sum())
        row['{} med_rem_days'.format(label)] = round(float(np.median(arr[mask, 0])), 1)
        row['{} P(90d)'.format(label)] = round(float(arr[mask, 2].mean()), 3)
    lines.append('\n{} by current status: {}'.format(model, row))

lines.append('\n' + '=' * 66)
lines.append('UPDATED CONCLUSIONS')
lines.append('=' * 66)
for title, header, text in conclusions:
    lines.append('\n--- {} ({}) ---'.format(title, header))
    lines.append(text)

with open('question2/forecasting_pilot_results.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Saved interesting cases + conclusions -> question2/forecasting_pilot_results.txt')