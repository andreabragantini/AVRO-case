# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:33:53 2020

@author: andre
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from tabulate import tabulate
import seaborn as sns
sns.set()

dataset = pd.read_csv('avro-issues.csv')

#%% Data Pre-Processing - Split Training Validation Sets
''' To calculate the required time to solve an alert we have to look only at
successfully solved alerts in the available dataset to be able to train our model.
This means we will look for Closed and Resolved ones.
1st appraoch: we could select only "Fixed" alerts.
2nd approach: we could select all alerts that has a resolution date'''

### 1st approach
fixed = dataset.query('resolution == "Fixed"').reset_index(drop=True)
fixed.columns
''' Required time to solve is the amount of time passing between the moment of
creation and the resolution date. '''

### 2nd approach
trainset = dataset[(~dataset["resolutiondate"].isnull()) & (~dataset["created"].isnull())]
trainset.shape

validationset = dataset[~dataset.index.isin(trainset.index)].reset_index(drop=True)
validationset.shape

# NB: from now we will call for simplicity the trainset as DATASET
dataset = trainset.reset_index(drop=True)

#%% Data Pre-Processing - Target Variable

### 1st approach:
fixed['resolutiondate']
# remove useless +0000 at the end
fixed['resolutiondate'] = fixed['resolutiondate'].str.replace('+000','',regex=False)
fixed['created'] = fixed['created'].str.replace('+000','',regex=False)

import datetime
fixed['resolutiondate'] = fixed['resolutiondate'].apply(datetime.datetime.strptime, args=('%Y-%m-%dT%H:%M:%S.%f',))
fixed['created'] = fixed['created'].apply(datetime.datetime.strptime, args=('%Y-%m-%dT%H:%M:%S.%f',))
# duration
fixed['duration'] = fixed['resolutiondate'] - fixed['created']

### Other way: from str to datetime
dataset.resolutiondate = pd.to_datetime(dataset['resolutiondate'])
dataset.updated = pd.to_datetime(dataset['updated'])
dataset.created = pd.to_datetime(dataset['created'])
# duration
dataset['duration'] = dataset['resolutiondate'] - dataset['created']
# from [ns] to [days]
dataset['duration'].astype('timedelta64[D]')


#%% Save datasets
'''After have created the target variable we can now save the datasets'''
# Directory creation
import os
if not os.path.exists('DataSets'):
    os.mkdir('DataSets')

# 1st appraoch - selected only fixed
fixed.to_csv('DataSets/dataset.csv', index=False)
# 2nd appraoch - selected with resolutiondate
dataset.to_csv('DataSets/dataset.csv', index=False)

validationset.to_csv('DataSets/validationset.csv', index=False)


#%% plot distribution of target variable
print('\nPlot Histogram for target variable:')
plt.figure(figsize=(12,8))
dataset['duration'].astype('timedelta64[D]').hist(bins=range(0,200,1))
plt.ylabel('N# of observations')
plt.xlabel('1st 200 Time classes [1 day]')
plt.savefig('ExploratoryAnalysis\histogram.png', bbox_inches='tight')
plt.show()

print('\nBoxPlot for target variable:')
plt.figure(figsize=(12,8))
dataset['duration'].astype('timedelta64[D]').plot.box()
plt.savefig('ExploratoryAnalysis\plotbox.png', bbox_inches='tight')
plt.show()

''' Almot all the observations are characterized by small resolution times [min,hours...}
but the distribution is highly skewed to the left and there are lots of outliers with
very "high" resolution times [months,years...]'''

#%% Data Preprocessing - Drop useless predictors
''' Following independent variables should not be considered in the learning model.
They do not offer useful information.
- project : same for all
- updated : updates are after the resolution date
- resolutiondate : already used for determining duration
- created : already used for determining duration
- key : different for all
- days_in_current_status : not relevant to analysis
'''

dropped = dataset.drop(['project','updated','created','resolutiondate','key', 'days_in_current_status'],axis=1)

#%% Check for missing values
dropped.isnull().values.any()
dropped.isnull().sum()

# Other: Impute NA to 0s
#data.fillna(0, inplace=True)


''' Notice how some description lengths are not available, we put 0 instead.'''
dropped['description_length'][dropped['description_length'].isna()] = 0

#%% Reporter VS Assignee
# Build confusion matrix
confusion_matrix = dataset.groupby(["assignee", "reporter"]).size().sort_values(ascending = False)[:30].unstack(fill_value=0)
confusion_matrix = confusion_matrix.reindex(confusion_matrix.sum().sort_values(ascending = False).index, axis=1)
cols = confusion_matrix.columns[:10]
confusion_matrix[cols]

''' OBSERVATION: Generally relevant reporters assign the issue themselves to be solved.
Also minor reporters (Other) assign the reported issue themselves.
Cutting is the main contributor and solves self-reported issues and reported issues
by minors (Other) contributors and other relevant contributors.'''

# What to do then with these 2 cat var?
# Should we hot-encode them and include in the model, or not?

''' With a very rough assumption we can say that all the reporters assign the
issues to themselves. I prefer to take reporters as cat var because it is always
present, even in new alerts to be forecasted. While sometimes happens that issues
are not assigned.'''

''' What if assignee can be usefull?
IDEA: assigned alerts --> SHORT resolution
      not assigned alerts --> LONG resolution
Unfortunately, it is not possible using a supervised learning ML technique to
find any observation that has not been assigned (assignee = NaN). 
In fact, all finished alerts have usually been ultimately assigned to be resolved.
There are cases of non-assigned solved alerts but those are specifically described
to be cases where the reporter is itself the assignee (leading presumably to a 
supposed shorter resolution of the "alert", so actually against this logic)
Therefore it is not possible to work on this feature with a supervised
learning method.

CONCLUSION: remove assignee'''

dropped = dropped.drop(['assignee'], axis=1)

#%% STATUS & RESOLUTION
# We have 2 cat vars left: 'status' and 'resolution'. What to do?
''' The scope of the exercize is to create a model which predicts the closing
time of a future alert based on available dataset.
Ideally, a new alert would have a "status" var that can assume the following:
Open, PatchAvailable, Reopened, but NEVER Closed or Resolved.
These last 2 classes belongs in fact to an "already resolved" alert.
The actual training set considers all alerts that have been closed/resolved 
for which it is possible to determine the duration/closingtime.
Therefore the var "status" should not be included in the model.
In fact, it determines the split in training/test sets.

At the same time, "resolution" classes have a value only for closed/resolved
alerts. Therefore new alerts cannot has these feature. Looking at current
available dataset it shows always "NaN".

CONCLUSION: We should remove "status" and "resolution" predictors.'''

set(validationset['resolution'])
set(validationset['status'])

dropped = dropped.drop(['status','resolution'], axis=1)

#%% Save Dataset
dropped.to_csv('DataSets/processed.csv', index=False)
