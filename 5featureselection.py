# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:02:05 2020

@author: andre
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

data = pd.read_csv('DataSets/encoded.csv')
data.columns

# convert duration data from str to timedelta64
#data['duration'] = pd.to_timedelta(data['duration'], unit='d')
# convert from timedelta64 to float64
#target = data['duration'] / np.timedelta64(1, 'h')

predictors = data.iloc[:,:-1]
predictors.columns

# create directory
import os
if not os.path.exists('FeatureSelection'):
    os.makedirs('FeatureSelection')

###############################################################################
#               FEATURES SELECTION
###############################################################################

y = data['duration']
x = data.iloc[:,:-1]
#x = data.iloc[:,6:13]

lr = LinearRegression()

''' PAY ATTENTION:
The Python packages EFS/SFS for feature selection include N Fold Cross Validation 
for both forward and backward. Cross-Validation errors are quantified with "scores".
It is often take the MSE to quantify the deviation from the validation set in CV.
Note: The Cross validation error (score) for SFS in Sklearn is negative, 
possibly because it computes the ‘neg_mean_squared_error’. 
In fact, all scorer objects follow the convention that higher return values are better 
than lower return values. Thus, metrics which measure the distance between 
the model and the data, like metrics.mean_squared_error, are available as 
neg_mean_squared_error which return the negated value of the metric.
I have taken the -ve of this neg_mean_squared_error. This should give mean_squared_error.
'''

##%% BEST FIT - Exhaustive search (EFS)
##Perform an Exhaustive Search
#efs1 = EFS(lr, 
#           min_features=1,
#           max_features=x.shape[1],
#           scoring='neg_mean_squared_error',
#           print_progress=True,
#           cv=5)
#
## Create a efs fit
#efs1 = efs1.fit(x.values, y.values)
#
## BEst fit steps with CrossValidation
#metrics = pd.DataFrame.from_dict(efs1.get_metric_dict(confidence_interval=0.90)).T
#print(metrics)
#
#
#print('Best negtive mean squared error: %.2f' % efs1.best_score_)
### Print the IDX of the best features 
#print('Best subset:', efs1.best_idx_)
#
## Features from forward fit
#b = list(efs1.best_idx_) 
#print("Features selected in forward fit")
#print(x.columns[b])

#%% FORWARD FIT - Sequential Search (SFS)
sfs_f = SFS(lr, 
          k_features=(1,predictors.shape[1]), 
          forward=True, # Forward fit
          floating=False, 
          scoring='neg_mean_squared_error',
          cv=5)

# Fit this on the data
sfs_f = sfs_f.fit(x.values, y.values)
# Get all the details of the forward fits
a=sfs_f.get_metric_dict()
n=[]
o=[]

# Compute the mean cross validation scores
for i in np.arange(1,predictors.shape[1]):
    n.append(-np.mean(a[i]['cv_scores']))  
m=np.arange(1,predictors.shape[1])

# Plot the CV scores vs the number of features
fig1=plt.plot(m,n,label='SFS_f')
plt.title('Mean CV Scores vs N# of features')
plt.xlabel('N# features')
plt.ylabel('MSE')

# Forward steps with Cross-Validation
metrics = pd.DataFrame.from_dict(sfs_f.get_metric_dict(confidence_interval=0.90)).T
print(metrics)

# Get the index of the minimum CV score
idx = np.argmin(n)
print("N# of features = %s" % idx)
#Get the features indices for the best forward fit and convert to list
b1=list(a[idx]['feature_idx'])
print(b1)

# Index the column names. 
# Features from forward fit
print("Features selected in forward fit")
print(x.columns[b1])

#%% BACKWARD FIT - Sequential Search (SBS)
# Create the SBS model
sfs_b = SFS(lr, 
          k_features=(1,predictors.shape[1]), 
          forward=False, # Backward
          floating=False, 
          scoring='neg_mean_squared_error',
          cv=5)

# Fit the model
sfs_b = sfs_b.fit(x.values, y.values)
a=sfs_b.get_metric_dict()
n=[]
o=[]

# Compute the mean of the validation scores
for i in np.arange(1,predictors.shape[1]):
    n.append(-np.mean(a[i]['cv_scores'])) 
m=np.arange(1,predictors.shape[1])

# Plot the Validation scores vs number of features
fig2=plt.plot(m,n,label='SFS_b')

# Backward steps with Cross-Validation
metrics = pd.DataFrame.from_dict(sfs_b.get_metric_dict(confidence_interval=0.90)).T
print(metrics)

# Get the index of minimum cross validation error
idx = np.argmin(n)
print("N# of features = %s" % idx)
#Get the features indices for the best backward fit and convert to list
b2=list(a[idx]['feature_idx'])
print(b2)

# Index the column names. 
# Features from backward fit
print("Features selected in backward fit")
print(x.columns[b2])

#%% Sequential Floating Forward Selection - SFFS
'''The Sequential Feature search also includes ‘floating’ variants which 
include or exclude features conditionally, once they were excluded or included. 
The SFFS can conditionally include features which were excluded from the previous step,
if it results in a better fit. 
This option will tend to a better solution, than plain simple SFS.'''

# Create the floating forward search
sffs = SFS(lr, 
          k_features=(1,predictors.shape[1]), 
          forward=True,  # Forward
          floating=True,  #Floating
          scoring='neg_mean_squared_error',
          cv=5)

# Fit a model
sffs = sffs.fit(x.values, y.values)
a=sffs.get_metric_dict()
n=[]
o=[]

# Compute mean validation scores
for i in np.arange(1,predictors.shape[1]):
    n.append(-np.mean(a[i]['cv_scores'])) 
m=np.arange(1,predictors.shape[1])

# Plot the cross validation score vs number of features
fig3=plt.plot(m,n, label='SFFS')

# Backward steps with Cross-Validation
metrics = pd.DataFrame.from_dict(sffs.get_metric_dict(confidence_interval=0.90)).T
print(metrics)

# Get the index of minimum cross validation error
idx = np.argmin(n)
print("N# of features = %s" % idx)
#Get the features indices for the best forward fit and convert to list
b3=list(a[idx]['feature_idx'])
print(b3)

# Index the column names. 
# Features from floating forward fit
print("Features selected in floating forward fit")
print(x.columns[b3])

#%% Sequential Floating Backward Selection - SFBS

# Create the floating backward search
sfbs = SFS(lr, 
          k_features=(1,predictors.shape[1]), 
          forward=False,  # Backward
          floating=True,  #Floating
          scoring='neg_mean_squared_error',
          cv=5)

# Fit a model
sfbs = sfbs.fit(x.values, y.values)
a=sfbs.get_metric_dict()
n=[]
o=[]

# Compute mean validation scores
for i in np.arange(1,predictors.shape[1]):
    n.append(-np.mean(a[i]['cv_scores'])) 
m=np.arange(1,predictors.shape[1])

# Plot the cross validation score vs number of features
fig3=plt.plot(m,n,label='SFBS')

# Backward steps with Cross-Validation
metrics = pd.DataFrame.from_dict(sffs.get_metric_dict(confidence_interval=0.90)).T
print(metrics)

# Get the index of minimum cross validation error
idx = np.argmin(n)
print("N# of features = %s" % idx)
#Get the features indices for the best backward fit and convert to list
b4=list(a[idx]['feature_idx'])
print(b4)

# Index the column names. 
# Features from flaoting backward fit
print("Features selected in floating backward fit")
print(x.columns[b4])

#%% Wrapping final results from SFS tecniques

# save the comparison plot with legend
plt.legend()
plt.savefig('FeatureSelection/CVscoresVSfeatures_comparison.png', bbox_inches='tight')

# selected features by different sequential searches
b1 # SFS_f
len(b1)
x.columns[b1]

b2 # SFS_b
len(b2)
x.columns[b2]

b3 # SFFS (floating)
len(b3)
x.columns[b3]

b4 # SFBS (floating)
len(b4)
x.columns[b4]

# Display results in array
selected= set(x.columns[b1]) | set(x.columns[b2]) | set(x.columns[b3]) | set(x.columns[b4])
l1 = [1 if i in set(x.columns[b1]) else 0 for i in selected ]
l2 = [1 if i in set(x.columns[b2]) else 0 for i in selected ]
l3 = [1 if i in set(x.columns[b3]) else 0 for i in selected ]
l4 = [1 if i in set(x.columns[b4]) else 0 for i in selected ]
columns = ['SFS_f','SFS_b','SFFS','SFBS']
data = np.array([l1,l2,l3,l4]).transpose()
results = pd.DataFrame(data=data, columns=columns, index=selected)
results



 