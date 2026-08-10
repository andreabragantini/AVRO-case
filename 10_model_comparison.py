# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 2026

@author: andre
MODEL COMPARISON
Final results step. Puts the three model families (regression, trees and
survival) side by side on the same held-out test set.

The forecasting pilot for the 324 still-open issues is NOT here anymore: it
lives in 9_predicting.py, which applies all five models to those issues. This
script only evaluates the models on the held-out 20% of the resolved issues.

All inputs are loaded from disk (models saved by scripts 6/7/8, evaluation
bundle saved by 8_survival_analysis.py), so this script is fully standalone.

Outputs (results/):
  - model_comparison.txt   all five models on the same test set
"""
import os
from joblib import load

import numpy as np

from sklearn.model_selection import train_test_split

from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import utils.avro_common as ac

print_section = ac.print_section

# ---------------------------------------------------------------------------
# 1. Load fitted models + evaluation bundle from disk
# ---------------------------------------------------------------------------
linreg = load('multi_lin_reg/linreg.joblib')
tree = load('regression_tree/tree.joblib')
rfr = load('regression_tree/rfr.joblib')
cph = load('survival_analysis/cph.joblib')
rsf = load('survival_analysis/rsf.joblib')

bundle = load('survival_analysis/survival_eval.joblib')
X = bundle['X']
raw = bundle['raw']
time_days = bundle['time_days']
event = bundle['event']
surv_test_idx = bundle['surv_test_idx']
y_test_days = bundle['y_test_days']

if not os.path.exists('results'):
    os.makedirs('results')

# ---------------------------------------------------------------------------
# 2. Cross-model comparison on the same held-out test set
# ---------------------------------------------------------------------------
print_section('Model comparison (regression + trees + survival)')

X_test = X.iloc[surv_test_idx]
days_actual = y_test_days

# Linear model was trained on the SFS-selected subset of the features.
X_lin = X_test.reindex(columns=linreg.feature_names_in_, fill_value=0)
lin_pred = np.exp(linreg.predict(X_lin)) / 1440.0
lin_r2 = linreg.score(X_lin, np.log(days_actual * 1440.0))

tree_pred = np.exp(tree.predict(X_test)) / 1440.0
tree_r2 = tree.score(X_test, np.log(days_actual * 1440.0))

rfr_pred = np.exp(rfr.predict(X_test)) / 1440.0
rfr_r2 = rfr.score(X_test, np.log(days_actual * 1440.0))

# Survival models: median survival time (days) on the same test rows.
cox_pred = np.asarray(cph.predict_median(X_test), dtype=float)
cox_pred[~np.isfinite(cox_pred)] = time_days.max()
rsf_sfs = rsf.predict_survival_function(X_test.to_numpy())
rsf_pred = np.array([
    np.interp(0.5, np.asarray(sf.y)[::-1], np.asarray(sf.x)[::-1]) for sf in rsf_sfs
])

comp_rows = [
    ('Linear regression (OLS)', ac.compute_model_metrics(days_actual, lin_pred, r2_log=lin_r2)),
    ('Decision tree', ac.compute_model_metrics(days_actual, tree_pred, r2_log=tree_r2)),
    ('Random forest', ac.compute_model_metrics(days_actual, rfr_pred, r2_log=rfr_r2)),
    ('Cox PH', ac.compute_model_metrics(days_actual, cox_pred)),
    ('Random survival forest', ac.compute_model_metrics(days_actual, rsf_pred)),
]
ac.write_metrics_table(
    'results/model_comparison.txt', comp_rows,
    title='Model comparison - all model families (test set, day scale)',
)

# Append the survival-specific, censored-aware C-index as a note. It uses a
# separate 80/20 split over ALL issues (including the censored ones), because
# the censored information is the main advantage of the survival models.
all_idx = np.arange(len(raw))
all_tr, all_te = train_test_split(all_idx, test_size=0.2, random_state=0)

cph_all = CoxPHFitter()
cox_df_all = X.iloc[all_tr].copy()
cox_df_all['time'] = time_days[all_tr]
cox_df_all['event'] = event[all_tr]
cph_all.fit(cox_df_all, duration_col='time', event_col='event')
cox_risk = cph_all.predict_partial_hazard(X.iloc[all_te]).to_numpy()

rsf_all = RandomSurvivalForest(n_estimators=200, random_state=42, n_jobs=1)
rsf_all.fit(X.iloc[all_tr].to_numpy(),
            Surv.from_arrays(event=event[all_tr].astype(bool), time=time_days[all_tr]))
rsf_risk = rsf_all.predict(X.iloc[all_te].to_numpy())

te_event = event[all_te].astype(bool)
te_time = time_days[all_te]
# both predict_partial_hazard (Cox) and rsf.predict (RSF) return risk scores
c_cox_cens = concordance_index_censored(te_event, te_time, cox_risk)[0]
c_rsf_cens = concordance_index_censored(te_event, te_time, rsf_risk)[0]
print('Censored-aware C-index (80/20 split over all 1458 issues): '
      'Cox {:.3f} | RSF {:.3f}'.format(c_cox_cens, c_rsf_cens))

with open('results/model_comparison.txt', 'a') as f:
    f.write('\nNote: survival models natively handle censored (still-open) issues.\n')
    f.write('      Censored-aware C-index on a separate 80/20 split over ALL issues\n')
    f.write('      (1134 resolved + 324 censored): Cox {:.3f} | RSF {:.3f}\n'.format(
        c_cox_cens, c_rsf_cens))
