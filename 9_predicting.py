# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:01:07 2020

@author: andre
QUESTION 2:
Implement the proposed model using the data and extract the predicted resolution 
time for 3 interesting cases.

NOTE: the rows in forecasting_pilot.csv are issues that were still OPEN at the time
of the snapshot, so they have no resolution date and no ground truth for the
target. This script is therefore a FORECASTING PILOT (a qualitative sanity
check on realistic, still-open cases), not a quantitative model evaluation.
Quantitative evaluation is done on a random 80/20 train/test split inside
6_multi_lin_reg.py and 7_regression_trees.py.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from avro_common import print_section

print_section('PREDICTING (FORECASTING PILOT)')

# The fitted models below come from the earlier modeling scripts
# (6_multi_lin_reg.py and 7_regression_trees.py), which run in the same
# shared session when invoked via run_analysis.py. If they are not present
# (e.g. running this file standalone), fail with a clear message instead of
# an obscure NameError.
linreg = globals().get('linreg')
tree = globals().get('tree')
rfr = globals().get('rfr')
if linreg is None or tree is None or rfr is None:
    raise RuntimeError(
        "This script must be run through run_analysis.py so it can reuse the "
        "models fitted by 6_multi_lin_reg.py and 7_regression_trees.py."
    )

forecasting_pilot = pd.read_csv('data_sets/forecasting_pilot.csv')
print('\nForecasting pilot on issues that were still open at snapshot time:')
print('  shape: {} rows x {} columns'.format(*forecasting_pilot.shape))
print('  statuses: {}'.format(sorted(forecasting_pilot['status'].unique())))
test = forecasting_pilot.copy()
test.columns

training = pd.read_csv('data_sets/encoded.csv')
training = training.drop('duration',axis=1)

# create directory
if not os.path.exists('question2'):
    os.makedirs('question2')

# take apart status
statuses = test['status']

# drop useless columns
test = test.drop(['project','updated','created','resolutiondate','key',
                  'days_in_current_status','assignee','resolution','status'],axis=1)

# Check for missing values
test.isnull().values.any()
test.isnull().sum()    
test.loc[test['description_length'].isna(), 'description_length'] = 0
    
#%% Reduce classes
# Reporter
counts = test['reporter'].value_counts()
selected = counts[counts > 10]
test['reporter'] = test['reporter'].map(lambda x: x if x in selected else 'Other')

# Issue_types
group1 = ['Bug', 'Improvement', 'Task', 'Test']
test['issue_type'] = test['issue_type'].map(lambda x : 'Short' if x in group1 else 'Long') 

#%% Log trasformation
test['vote_count'] = np.log(test['vote_count']+1)
# comment_count
test['comment_count'] = np.log(test['comment_count']+1)
# description_length
test['description_length'] = np.log(test['description_length']+1)
# watch-count
test['watch_count'] = np.log(test['watch_count']+1)

#%% Encoding
cat_vars = ['priority','issue_type','reporter']
one_hot_test = pd.get_dummies(test[cat_vars],drop_first=True)
one_hot_test.columns
test = pd.concat([test, one_hot_test], axis=1)
test = test.drop(['reporter','issue_type','priority'], axis = 1)
test.columns

# fill missing columns in encoded_test set with columns from encoded_training
training.columns
newcolumns = [x for x in training.columns if x not in test.columns]
for col in newcolumns:
    test['{}'.format(col)] = 0

# Align each forecasting-pilot matrix to the feature set used by each fitted model.
test_full = test.reindex(columns=training.columns, fill_value=0)
test_linreg = test_full.reindex(columns=linreg.feature_names_in_, fill_value=0)

#%% Prediction
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

# from float64 to timedelta64
df_pred['LinearModel'] = pd.to_timedelta(df_pred['LinearModel'], unit='m')
df_pred['RegrTree'] = pd.to_timedelta(df_pred['RegrTree'], unit='m')
df_pred['RandomForest'] = pd.to_timedelta(df_pred['RandomForest'], unit='m')

print('\nPredicted resolution times for the discussed forecasting cases (days):')
interesting = df_pred.loc[[321, 120, 235, 231, 77, 243, 113]]
print((interesting.astype('timedelta64[h]') / 24).round(1).to_string())


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

