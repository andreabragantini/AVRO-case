# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:39:52 2020

@author: andre
"""
import pandas as pd
from avro_common import print_section

print_section('4. ENCODING')

#data = pd.read_csv('data_sets/processed.csv')
data = pd.read_csv('data_sets/transformed_nonencoded.csv')
data.columns

targetvar = pd.to_timedelta(data['duration']).dt.total_seconds() / 86400

#############################################################################
#           One-Hot Encoding of categorical variables
###############################################################################

# Cat Var with current levels:
set1 = set(data['priority'])
set2 = set(data['issue_type'])
set3 = set(data['reporter'])


''' There are a lot of levels for each cat var in the dataset, therefore we
should reduce the levels by using combining methods and then use dummy encoding.'''

# Dummy Encoding
# NOTE: pd.get_dummies(drop_first=True) is used so that the dummy column names
# (e.g. reporter_cutting, priority_Minor) match the ones produced on the
# forecasting-pilot set in 9_predicting.py. The previous manual loop named
# columns by index (priority_0, reporter_1, ...) which never matched the
# forecasting-pilot encoding, so every categorical feature was silently zeroed out
# in the forecasting pilot. The manual loop is kept below for reference.
data = pd.get_dummies(data, columns=['priority', 'issue_type', 'reporter'], drop_first=True)

# # Manual one-hot encoding (reference only - kept but not used):
# # priority
# l = list(set1)
# for i, level in enumerate(l[:-1]):
#     data['priority_{}'.format(i)] = 0
#     boolean = data['priority'] == level
#     data.loc[boolean, 'priority_{}'.format(i)] = 1
# data = data.drop('priority',axis = 1)
#
# # issue_type
# l = list(set2)
# for i, level in enumerate(l[:-1]):
#     data['issue_type_{}'.format(i)] = 0
#     boolean = data['issue_type'] == level
#     data.loc[boolean, 'issue_type_{}'.format(i)] = 1
# data = data.drop('issue_type',axis = 1)
#
# # reporter
# l = list(set3)
# for i, level in enumerate(l[:-1]):
#     data['reporter_{}'.format(i)] = 0
#     boolean = data['reporter'] == level
#     data.loc[boolean, 'reporter_{}'.format(i)] = 1
# data = data.drop('reporter',axis = 1)

data.columns

#%% Move target variable to the end of dataframe
data = data[[c for c in data if c not in ['duration']] + ['duration']]

#%% Save encoded dataset
data.to_csv('data_sets/encoded.csv', index=False)
print('\nSaved one-hot encoded dataset to data_sets/encoded.csv')


#%% Other methods
# get_dummies
# Kept as a reference implementation only. The main pipeline above already
# saves the encoded dataset, so this alternative block is skipped unless the
# categorical columns are still present.
cat_vars = ['priority','issue_type','reporter']
if set(cat_vars).issubset(data.columns):
    one_hot_data = pd.get_dummies(data[cat_vars], drop_first=True)
    one_hot_data.columns
    data = pd.concat([data, one_hot_data], axis=1)
    data = data.drop(['reporter','issue_type','priority'], axis = 1)
    data.columns
