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
    long the issue has already been open) plus P(resolve in 30/90/180 days)

The fitted models are loaded from disk: scripts 6/7/8 saved them (and the
out-of-sample pilot survival models) into their directories, so this script is
fully standalone.
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
# The legacy hand-rolled pipeline in this script used test-set reporter counts,
# which disagreed with the survival training encoding.
test_full = prepare_forecasting_pilot_features(forecasting_pilot, training.columns)
test_linreg = test_full.reindex(columns=linreg.feature_names_in_, fill_value=0)

#%% Prediction - regression / trees
lr_pred = linreg.predict(test_linreg)    # multivariate linear model
tree_pred = tree.predict(test_full)      # regression tree model
rf_pred = rfr.predict(test_full)         # random forest regression model

# comparison
df_pred=pd.DataFrame({'LinearModel':lr_pred, 'RegrTree':tree_pred, 'RandomForest':rf_pred})

# Plots
df_pred.plot(figsize=(12,5),marker='.')
plt.xlabel("Forecasting pilot observations",fontsize=15)
plt.ylabel("LOG(Duration)",fontsize=15)
plt.title("Forecasting Pilot Prediction - trasformed",fontsize=18)
plt.savefig('question2/predictionComparison_log.png')
plt.close()

df_pred = np.exp(df_pred)
df_pred.plot(figsize=(12,5),marker='.')
plt.xlabel("Forecasting pilot observations",fontsize=15)
plt.ylabel("Duration",fontsize=15)
plt.title("Forecasting Pilot Prediction",fontsize=18)
plt.savefig('question2/predictionComparison.png')
plt.close()

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
pilot = pd.DataFrame({
    'key': forecasting_pilot['key'],
    'status': forecasting_pilot['status'],
    'elapsed_days': np.round(elapsed, 1),
    'LinearModel_total_days': np.round(np.exp(lr_pred) / 1440, 1),
    'RegrTree_total_days': np.round(np.exp(tree_pred) / 1440, 1),
    'RandomForest_total_days': np.round(np.exp(rf_pred) / 1440, 1),
    'cox_median_remaining': np.round(cox_pilot[:, 0], 1),
    'cox_P_30d': np.round(cox_pilot[:, 1], 3),
    'cox_P_90d': np.round(cox_pilot[:, 2], 3),
    'cox_P_180d': np.round(cox_pilot[:, 3], 3),
    'rsf_median_remaining': np.round(rsf_pilot[:, 0], 1),
    'rsf_P_30d': np.round(rsf_pilot[:, 1], 3),
    'rsf_P_90d': np.round(rsf_pilot[:, 2], 3),
    'rsf_P_180d': np.round(rsf_pilot[:, 3], 3),
})
pilot.to_csv('question2/forecasting_pilot_predictions.csv', index=False)
print('\nSaved unified forecasting-pilot predictions -> question2/forecasting_pilot_predictions.csv')

print('\nAverage survival forecast for the 324 open issues:')
summary = pd.DataFrame({
    'median_remaining_days': [np.median(cox_pilot[:, 0]), np.median(rsf_pilot[:, 0])],
    'P(resolve in 30d)': [cox_pilot[:, 1].mean(), rsf_pilot[:, 1].mean()],
    'P(resolve in 90d)': [cox_pilot[:, 2].mean(), rsf_pilot[:, 2].mean()],
    'P(resolve in 180d)': [cox_pilot[:, 3].mean(), rsf_pilot[:, 3].mean()],
}, index=['Cox PH', 'Random survival forest']).round(3)
print(summary.to_string())

# Status split: do issues currently "Patch Available" get a shorter horizon?
print('\nBy current status (Patch Available vs other open statuses):')
for model, arr in [('Cox PH', cox_pilot), ('RSF', rsf_pilot)]:
    is_patch = (pilot['status'] == 'Patch Available').to_numpy()
    row = {}
    for label, mask in [('Patch Available', is_patch), ('Other open', ~is_patch)]:
        row['{} n'.format(label)] = int(mask.sum())
        row['{} med_rem_days'.format(label)] = round(float(np.median(arr[mask, 0])), 1)
        row['{} P(90d)'.format(label)] = round(float(arr[mask, 2].mean()), 3)
    print('  {}: {}'.format(model, row))

#%% Interesting cases (day scale, all five models)
print('\nPredicted resolution times for the discussed forecasting cases:')
df_pred['LinearModel'] = pd.to_timedelta(df_pred['LinearModel'], unit='m')
df_pred['RegrTree'] = pd.to_timedelta(df_pred['RegrTree'], unit='m')
df_pred['RandomForest'] = pd.to_timedelta(df_pred['RandomForest'], unit='m')

interesting = df_pred.loc[[321, 120, 235, 231, 77, 243, 113]]
interesting = (interesting.astype('timedelta64[h]') / 24).round(1)
interesting.columns = ['LinearModel_total_days', 'RegrTree_total_days', 'RandomForest_total_days']
# cox_pilot/rsf_pilot rows align with forecasting_pilot rows, so use position
# directly (the interesting cases are forecasting_pilot rows 321, 120, ...).
interesting['Cox_median_remaining'] = cox_pilot[[321, 120, 235, 231, 77, 243, 113], 0].round(1)
interesting['RSF_median_remaining'] = rsf_pilot[[321, 120, 235, 231, 77, 243, 113], 0].round(1)
print(interesting.to_string())

#%% 3 Interesting Cases
test = pd.read_csv('data_sets/forecasting_pilot.csv')

df_pred.loc[321,:]
test.loc[321,:]
''' This alert has an "Open" status, has "Major" priority and as of "NewFeature" issue type.
Assignee and reporter are the same non-frequent contributor. 
It has a high number of votes, comments 
and watchers sign of great interest from the comunity.
The linear regression returns an "explosive" predictions but however also the tree
methods predicts quite a long resolution times.
In fact, looking back at the original dataset, therefore an information not
processed by my models, this alert have been in the Open status for almost 2 years.
This might sound as the issue is not really proceeding and might remain like so for much longer.'''

df_pred.loc[120,:]
test.loc[120,:]
''' This alert has an "PatchAvailable" status, has "Major" priority and as of "NewFeature" issue type.
It has not been assigned yet, which is normally a sign of longer times,
but there is already an available solution. Probably cutting is taking care of it.
Also this thread is pretty popular on the website as it has lots of comments,
watchers and votes. The status suggests that it is going to be solved possibly soon.
However, the issue seems stuck in the same status for almost 2 years without progressing.
This behaviour again pushes the linear regression to return an "inflate" results
while the trees method are more optimistic, predicting a more or less close resolution.''' 

df_pred.loc[235,:]
test.loc[235,:]
''' This alert has an "Open" status, has "Major" priority and as of "Improvement" issue type.
Assignee and reporter are the same frequent contributor (hammer). 
It has a medium number of votes, comments and watchers.
It is stuck in this status without solutions for more than 3 years, in my humble opinion
it is hard that it is going to be solved any sooner.
The linear model return a completely wrong results of 50 days while tree methods
advices a longer, although not huge, resolution times. Surprisingly in this case
a more reliable estimate is given by the single tree.'''


df_pred.loc[231,:]
test.loc[231,:]
''' same as above but linear model is less wrong'''


df_pred.loc[77,:]
test.loc[77,:]
''' This alert has an "Open" status, has "Major" priority and as of "Improvement" issue type.
It has not been assigned yet, which is already a sign of longer times.
In this case my models all predict very short resolution times.
Given the fact that the issue is left unassigned for almost 4 years, those
prediction are surely wrong.'''

df_pred.loc[243,:]
test.loc[243,:]
''' same as above'''


df_pred.loc[113,:]
test.loc[113,:]
''' This alert has an "PatchAvailable" status, has "Major" priority and as of "Bug" issue type.
Assignee and reporter are the same frequent contributor (tonwhite), possible sign 
of a quick resolution of the issue.
The model is in this status only since 5 days and there is already a Patch Available. 
Everything seems going for a quick resolution of the issue.
In fact, all models return quite low predicted resolution times.'''
