# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 2026

@author: andre
SURVIVAL ANALYSIS
Predict the time-to-resolution of an Avro issue with survival models (Cox
proportional hazards and random survival forest) so that we can also exploit
the 324 issues that were still OPEN at snapshot time as right-censored
observations, instead of discarding them like the regression/tree approaches.

Outputs (survival_analysis/):
  - km_overall.png / km_by_issue_type.png  Kaplan-Meier survival curves
  - model_metrics.txt      Cox + RSF metrics (same format as the other dirs)

The cross-model comparison and the forecasting-pilot survival forecast are in
10_model_comparison.py (which runs after this script in the shared session).

The script is self-contained: it reads data_sets/raw/avro-issues-merged.csv and
rebuilds the exact 32-feature encoding used by data_sets/encoded.csv.
"""
import os
from joblib import dump

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

import avro_common as ac

print_section = ac.print_section

print_section('SURVIVAL ANALYSIS')

if not os.path.exists('survival_analysis'):
    os.makedirs('survival_analysis')

# ---------------------------------------------------------------------------
# 1. Build the survival dataset (time in days, event flag, 32 features)
# ---------------------------------------------------------------------------
raw = pd.read_csv('data_sets/raw/avro-issues-merged.csv')
print('Loaded {} issues from data_sets/raw/avro-issues-merged.csv'.format(len(raw)))

created = pd.to_datetime(raw['created'], errors='coerce', utc=True).dt.tz_localize(None)
resolutiondate = pd.to_datetime(raw['resolutiondate'], errors='coerce', utc=True).dt.tz_localize(None)
updated = pd.to_datetime(raw['updated'], errors='coerce', utc=True).dt.tz_localize(None)

resolved = resolutiondate.notna().to_numpy()

# time to event (days): resolution date for resolved issues, last update for
# the still-open (censored) ones.
time_days = np.where(
    resolved,
    (resolutiondate - created).dt.total_seconds().to_numpy() / 86400.0,
    (updated - created).dt.total_seconds().to_numpy() / 86400.0,
)
event = resolved.astype(int)

print('  resolved (event=1): {} | censored/open (event=0): {}'.format(
    int(event.sum()), int((event == 0).sum())))
print('  time-to-event range: {:.2f} - {:.0f} days (median {:.1f})'.format(
    time_days.min(), time_days.max(), np.median(time_days)))

# Rebuild the same 32-feature encoding as data_sets/encoded.csv.
frame = raw.copy()
frame['description_length'] = frame['description_length'].fillna(0)
frame = ac.reduce_categories(frame)

# Keep the reduced issue_type group (Short/Long) for the KM stratification.
issue_type_group = frame['issue_type'].to_numpy()

frame = ac.log_transform_numeric_features(frame)
frame = frame.drop(columns=['vote_count', 'comment_count', 'watch_count'])
frame = pd.get_dummies(frame, columns=['priority', 'issue_type', 'reporter'], drop_first=True)

encoded_cols = [c for c in pd.read_csv('data_sets/encoded.csv').columns if c != 'duration']
X = frame[encoded_cols].copy()
assert list(X.columns) == encoded_cols, 'Feature mismatch with data_sets/encoded.csv'
print('  feature matrix: {} rows x {} columns (parity with encoded.csv)'.format(*X.shape))

# ---------------------------------------------------------------------------
# 2. Same 80/20 split on the resolved issues as the regression/tree scripts
# ---------------------------------------------------------------------------
resolved_idx = np.flatnonzero(resolved)
tr_idx, te_idx = train_test_split(resolved_idx, test_size=0.2, random_state=0)

# Regression/trees train on the resolved train only. Survival can additionally
# use the censored (open) issues, so we train on resolved-train + all censored.
censored_idx = np.flatnonzero(~resolved)
surv_train_idx = np.concatenate([tr_idx, censored_idx])
surv_test_idx = te_idx

y_test_days = time_days[surv_test_idx]

# ---------------------------------------------------------------------------
# 3. Kaplan-Meier curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
km_t, km_s = kaplan_meier_estimator(event.astype(bool), time_days)
ax.step(km_t, km_s, where='post')
ax.set_title('Kaplan-Meier: probability issue is still open', fontsize=14)
ax.set_xlabel('Days since creation', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.savefig('survival_analysis/km_overall.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for group in ['Short', 'Long']:
    mask = (issue_type_group == group) & event.astype(bool)
    g_t, g_s = kaplan_meier_estimator(np.ones(mask.sum(), dtype=bool), time_days[mask])
    ax.step(g_t, g_s, where='post', label='{} issues (n={})'.format(group, mask.sum()))
ax.set_title('Kaplan-Meier by issue_type group', fontsize=14)
ax.set_xlabel('Days since creation', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('survival_analysis/km_by_issue_type.png')
plt.close()
print('\nSaved Kaplan-Meier curves to survival_analysis/.')

# ---------------------------------------------------------------------------
# 4. Cox proportional hazards model
# ---------------------------------------------------------------------------
print_section('Cox proportional hazards (lifelines)')
cox_df = X.iloc[surv_train_idx].copy()
cox_df['time'] = time_days[surv_train_idx]
cox_df['event'] = event[surv_train_idx]

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time', event_col='event')
cph.print_summary()

cox_pred = np.asarray(cph.predict_median(X.iloc[surv_test_idx]), dtype=float)
cox_pred[~np.isfinite(cox_pred)] = time_days.max()  # never reaches 0.5 -> very long
cox_c_index_test = concordance_index_censored(
    np.ones(len(y_test_days), dtype=bool), y_test_days, -cox_pred)[0]
print('C-index (test, day scale): {:.3f}'.format(cox_c_index_test))

# persist the fitted models so the comparing script can run standalone
dump(cph, 'survival_analysis/cph.joblib')

# ---------------------------------------------------------------------------
# 5. Random survival forest
# ---------------------------------------------------------------------------
print_section('Random survival forest (scikit-survival)')
rsf = RandomSurvivalForest(n_estimators=200, random_state=42, n_jobs=1)
rsf.fit(X.iloc[surv_train_idx].to_numpy(),
        Surv.from_arrays(event=event[surv_train_idx].astype(bool),
                         time=time_days[surv_train_idx]))
# rsf.predict() returns a RISK score (higher = shorter survival); for the
# day-scale metrics we need the median survival time instead.
rsf_sfs = rsf.predict_survival_function(X.iloc[surv_test_idx].to_numpy())
rsf_pred = np.array([
    np.interp(0.5, np.asarray(sf.y)[::-1], np.asarray(sf.x)[::-1]) for sf in rsf_sfs
])
rsf_c_index_test = concordance_index_censored(
    np.ones(len(y_test_days), dtype=bool), y_test_days, -rsf_pred)[0]
print('C-index (test, day scale): {:.3f}'.format(rsf_c_index_test))

# persist the fitted models so the comparing script can run standalone
dump(rsf, 'survival_analysis/rsf.joblib')

# ---------------------------------------------------------------------------
# 6. Model metrics table (same format as multi_lin_reg/ and regression_tree/)
# ---------------------------------------------------------------------------
surv_rows = [
    ('Cox PH', ac.compute_model_metrics(y_test_days, cox_pred)),
    ('Random survival forest', ac.compute_model_metrics(y_test_days, rsf_pred)),
]
ac.write_metrics_table(
    'survival_analysis/model_metrics.txt', surv_rows,
    title='Survival models - model metrics (test set, day scale)',
)

# ---------------------------------------------------------------------------
# 7. Out-of-sample pilot models + evaluation bundle (for scripts 9 and 10)
# ---------------------------------------------------------------------------
# The forecasting pilot in 9_predicting.py predicts the 324 still-open issues.
# For that forecast to be out-of-sample, the survival models must be trained on
# the RESOLVED issues only (the cph/rsf above include the censored issues in
# training). Fit and persist those pilot models here so script 9 is standalone.
pilot_fit_idx = tr_idx
cox_p = CoxPHFitter()
cox_df_p = X.iloc[pilot_fit_idx].copy()
cox_df_p['time'] = time_days[pilot_fit_idx]
cox_df_p['event'] = event[pilot_fit_idx]
cox_p.fit(cox_df_p, duration_col='time', event_col='event')
dump(cox_p, 'survival_analysis/cox_pilot.joblib')

rsf_p = RandomSurvivalForest(n_estimators=200, random_state=42, n_jobs=1)
rsf_p.fit(X.iloc[pilot_fit_idx].to_numpy(),
          Surv.from_arrays(event=event[pilot_fit_idx].astype(bool),
                           time=time_days[pilot_fit_idx]))
dump(rsf_p, 'survival_analysis/rsf_pilot.joblib')
print('\nSaved pilot survival models -> survival_analysis/cox_pilot.joblib, rsf_pilot.joblib')

# Evaluation bundle for 10_model_comparison.py (the shared-session globals it
# used to read are replaced by this file, so the script runs standalone).
dump({
    'X': X,
    'raw': raw,
    'time_days': time_days,
    'event': event,
    'tr_idx': tr_idx,
    'censored_idx': censored_idx,
    'surv_test_idx': surv_test_idx,
    'y_test_days': y_test_days,
}, 'survival_analysis/survival_eval.joblib')
print('Saved survival evaluation bundle -> survival_analysis/survival_eval.joblib')

print('\nSurvival models fitted. The forecasting pilot for the 324 still-open '
      'issues is produced by 9_predicting.py and the cross-model comparison by '
      '10_model_comparison.py.')
