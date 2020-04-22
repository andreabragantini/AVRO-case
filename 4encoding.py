# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:39:52 2020

@author: andre
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('DataSets/processed.csv')
data = pd.read_csv('DataSets/trasformed_nonencoded.csv')
data.columns

targetvar = pd.to_timedelta(data['duration']).astype('timedelta64[D]')

#############################################################################
#           One-Hot Encoding of categorical variables
###############################################################################

#%% Manual One-Hot Encoding

# Cat Var with current levels:
#1'priority'
set1 = set(data['priority'])
#2'issue_type'
set2 = set(data['issue_type'])
#3'reporter'
set3 = set(data['reporter'])


''' There are a lot of levels for each cat var in the dataset, therefore we
should reduce the levels by using combining methods and then use dummy encoding.'''

# Dummy Encoding

# priority
l = list(set1)
for i, level in enumerate(l[:-1]):
    data['priority_{}'.format(i)] = 0
    boolean = data['priority'] == level
    data['priority_{}'.format(i)][boolean] = 1
data = data.drop('priority',axis = 1)

# issue_type
l = list(set2)
for i, level in enumerate(l[:-1]):
    data['issue_type_{}'.format(i)] = 0
    boolean = data['issue_type'] == level
    data['issue_type_{}'.format(i)][boolean] = 1
data = data.drop('issue_type',axis = 1)

# reporter
l = list(set3)
for i, level in enumerate(l[:-1]):
    data['reporter_{}'.format(i)] = 0
    boolean = data['reporter'] == level
    data['reporter_{}'.format(i)][boolean] = 1
data = data.drop('reporter',axis = 1)

data.columns

#%% Move target variable to the end of dataframe
data = data[[c for c in data if c not in ['duration']] + ['duration']]

#%% Save encoded dataset
data.to_csv('DataSets/encoded.csv', index=False)


#%% Other methods
# get_dummies
cat_vars = ['priority','issue_type','reporter']
one_hot_data = pd.get_dummies(data[cat_vars],drop_first=True)
one_hot_data.columns
data = pd.concat([data, one_hot_data], axis=1)
data = data.drop(['reporter','issue_type','priority'], axis = 1)
data.columns
