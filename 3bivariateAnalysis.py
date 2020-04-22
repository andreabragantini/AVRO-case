# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:42:32 2020

@author: andre
BIVARIATE ANALYSIS
In this script a bivariate analysis is performed to better study the relation
between our target variable ("duration") and its numerical or categorical predictors.
NB: This means that we are looking only at our new processed dataset of 
closed/resolved issues for the training phase-
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('DataSets/processed.csv')
data.duration= pd.to_timedelta(data['duration']).astype('timedelta64[m]')
data.columns
cat_vars = [ col for col in data.columns if data[col].dtype.kind not in 'biufcm']
cat_vars.remove('reporter')
num_vars = [ col for col in data.columns if data[col].dtype.kind in 'biufc' ]
num_vars.remove('duration')
target = data.duration

# NB The meaning of biufc: b bool, i int (signed), u unsigned int, f float, c complex, m timedelta

# create directories
import os
if not os.path.exists('BivariateAnalysis/Numerical'):
    os.makedirs('BivariateAnalysis/Numerical')
if not os.path.exists('BivariateAnalysis/FullClasses'):
    os.makedirs('BivariateAnalysis/FullClasses')
if not os.path.exists('BivariateAnalysis/ReducedClasses'):
    os.makedirs('BivariateAnalysis/ReducedClasses')

#%% Target Variable VS Categorical Predictors - FullClasses

subset = data #data[data["closing_time"] < 60]
for c in cat_vars:
    x = subset['duration'].values
    y = subset[c].values
        
    confusion_matrix = subset.groupby(['duration', c]).size().sort_values(ascending = False).unstack(fill_value=0)
    confusion_matrix = confusion_matrix.reindex(confusion_matrix.sum().sort_values(ascending = False).index, axis=1)
    
    confusion_matrix = confusion_matrix / confusion_matrix.sum()
    print("####################################################################################################")
    print("\n\033[1m" + "Analisi Bivariata closing_time" + "-" + c + "\033[0;0m \n")

    confusion_matrix.plot.line(title = "Analisi Bivariata duration" + "-" + c, figsize= (8,8))
    plt.savefig('BivariateAnalysis/FullClasses/durationVS{}.png'.format(c))
    data.boxplot(column="duration",by= c, figsize= (8,8)) 
    plt.savefig('BivariateAnalysis/FullClasses/durationVS{}_box.png'.format(c))
    plt.show()
    print("####################################################################################################")

#%% Reduce Classes - Combine Methods
''' The objective is to try to reduce the many levels present in our categorical vars'''

# Reporter
data['reporter'].value_counts()[:40].plot(kind='bar')
plt.title('Most frequent Reporters - Training Set')
plt.tight_layout()
plt.savefig('ExploratoryAnalysis\Reporters.png', bbox_inches='tight')
plt.show()
# Only reporters with more than 10 counts are considered
counts = data['reporter'].value_counts()
selected = counts[counts > 10]
data['reporter'] = data['reporter'].map(lambda x: x if x in selected else 'Other')

# Issue Type
''' It has been observed in Bivariate Analysis with Full Classes that on average 
issue_type leves as Bug, Improvement, Task, Test tend to have SHORT resolution times.
On the other hand, levels as NewFeature, Subtask and Wish tend to have LONG resolution times.
Therefore we try to encode this two different groups'''
plt.barh(data['issue_type'].value_counts().index,data['issue_type'].value_counts())
plt.title('Issue Types - Training Set')
plt.tight_layout()
plt.savefig('ExploratoryAnalysis\Issue_types.png', bbox_inches='tight')
plt.show()
# Create the 2 levels for issue_types
group1 = ['Bug', 'Improvement', 'Task', 'Test']
data['issue_type'] = data['issue_type'].map(lambda x : 'Short' if x in group1 else 'Long') 

## Priority
#''' Here the human istinct should bring to the conclusion that presumably the 
#priority levels (Majorm, Critical, Blocker) tend to have SHORT resolution times
#and on the other hand (Minor,Trivial) tend to have LONG resolution times. 
#Unfortunately, distribution of data does NOT show that (see chart above). 
#In fact all the levels have on average low resolution times and the presence of 
#many outliers with high resolution times'''
#plt.bar(data['priority'].value_counts().index,data['priority'].value_counts())
#plt.title('Priorities - Training Set')
#plt.tight_layout()
#plt.savefig('ExploratoryAnalysis\Priorities.png', bbox_inches='tight')
#plt.show()
## Only frequent priority types are considered 
## This way we end up with 3 levels: Major, Minor, Other
#counts = data['priority'].value_counts()
#others = counts[counts < 50]
#data['priority'] = data['priority'].map(lambda x : 'Others' if x in others else x) 
# Other way, wee keep only 2 classes (Major, Minor) with the same intuitive logic as before
#major = ['Major','Critical','Blocker']          # supposed to be SHORT
#minor = ['Minor','Trivial']                     # supposed to be LONG
#data['priority'] = data['priority'].map(lambda x : 'Major' if x in major else 'Minor') 


# New levels of cat vars:
#1'priority'
set1 = set(data['priority'])
#2'issue_type'
set2 = set(data['issue_type'])
#3'reporter'   
set3 = set(data['reporter'])
    
#%% Target Variable VS Categorical Predictors - ReducedClasses
cat_vars.append('reporter')

subset = data #data[data["closing_time"] < 60]
for c in cat_vars:
    x = subset['duration'].values
    y = subset[c].values
        
    confusion_matrix = subset.groupby(['duration', c]).size().sort_values(ascending = False).unstack(fill_value=0)
    confusion_matrix = confusion_matrix.reindex(confusion_matrix.sum().sort_values(ascending = False).index, axis=1)
    
    confusion_matrix = confusion_matrix / confusion_matrix.sum()
    print("####################################################################################################")
    print("\n\033[1m" + "Analisi Bivariata closing_time" + "-" + c + "\033[0;0m \n")

    confusion_matrix.plot.line(title = "Analisi Bivariata duration" + "-" + c, figsize= (8,8))
    plt.savefig('BivariateAnalysis/ReducedClasses/durationVS{}.png'.format(c))
    data.boxplot(column="duration",by= c, figsize= (8,8), rot=45) 
    plt.savefig('BivariateAnalysis/ReducedClasses/durationVS{}_box.png'.format(c))
    plt.show()
    print("####################################################################################################")

#%% Target Variable VS Numerical Predictors

# simple scatter plots 
for c in num_vars:
    plt.figure(figsize=(12,8))
    plt.title("{} vs. duration".format(c),fontsize=16)
    plt.scatter(x=data[c],y=target,color='blue',edgecolor='k')
    plt.grid(True)
    plt.xlabel(c,fontsize=14)
    plt.ylabel('Alert Duration [D]',fontsize=14)
    plt.savefig('BivariateAnalysis/Numerical/durationVS{}'.format(c))
    plt.show()

## hexagonal plots
#for i in range(0,len(num_vars)):
#    
#    #plt.figure(figsize=(12,8))
#    #data.plot.hexbin(x='duration', y= num_vars[i], gridsize=15, sharex= False)
#    lm = sns.jointplot(x='duration', y=num_vars[i], data=data, kind='hex', gridsize=15)
#    
#    # Access the Figure
#    fig = lm.fig 
#
#    # Add a title to the Figure
#    fig.suptitle('duration vs ' + num_vars[i], fontsize=12)
#    
#    # save fig
#    plt.savefig('BivariateAnalysis/Numerical/durationVS{}_Hex'.format(num_vars[i]))
#    # Show the plot
#    plt.show()
    
#%% Pairplots
from seaborn import pairplot

num_vars.append('duration')
pairplot(data[num_vars])    
plt.title('Pairplot for numerical features')
plt.savefig('BivariateAnalysis/Numerical/pairplot_num_vars.png')

#%% Multi-Collinearity check between numerical predictors

from statsmodels.graphics.correlation import plot_corr

num_vars.remove('duration')
corr = data[num_vars].corr()
corr

fig = plot_corr(corr,xnames=corr.columns)
plt.savefig('BivariateAnalysis/Numerical/heatmap.png')

''' The heatmap shows some correlation between 'watch_count' and both
'vote_count' and 'comment_count'. These last two are also a bit correlated.'''

#%% DATA TRASFORMATION - LOGARITHMIC
''' some predictors and the target variable present a very skewed distribution.
THerefore we should consider to apply the logarithmic transformation.
This helps in turning the distribution is something more gaussian.
Let's apply a log-log trasformation '''

import numpy as np
from statsmodels.stats.stattools import jarque_bera as jb
from statsmodels.stats.stattools import omni_normtest as omb
from statsmodels.compat import lzip

# Jarque-Bera normality test 
name = ['Jarque-Bera', 'Chi^2 two-tail probability', 'Skewness', 'Kurtosis']
test_results = jb(data.duration)
lzip(name, test_results)


# vote_count
data.vote_count = np.log(data.vote_count+1)
# comment_count
data.comment_count = np.log(data.comment_count+1)
# description_length
data.description_length = np.log(data.description_length+1)
# watch-count
data.watch_count = np.log(data.watch_count+1)
# duration
data.duration = np.log(data.duration)

# run test again
test_results = jb(data.duration)
lzip(name, test_results)                        # very improved! :)


# Pairplot transformed
pairplot(data[num_vars])    
plt.title('Pairplot for trasformed numerical features')
plt.savefig('BivariateAnalysis/Numerical/pairplot_num_vars_log.png')

# trasformed duration (target variable)
#plt.figure(figsize=(12,8))
data['duration'].hist()
plt.ylabel('N# of observations')
plt.xlabel('Log(Time)')
plt.title('Log-trasformed target variable')
plt.savefig('ExploratoryAnalysis\duration_log.png', bbox_inches='tight')
plt.show()


#%% Save trasformed dataset
data.to_csv('DataSets/trasformed_nonencoded.csv', index=False)

#%% Repeat Bivariate Analysis on trasformed data

### Target Variable VS Numerical Predictors
# simple scatter plots 
for c in num_vars:
    #plt.figure(figsize=(12,8))
    plt.title("{} vs. duration (transformed)".format(c),fontsize=16)
    plt.scatter(x=data[c],y=data.duration,color='blue',edgecolor='k')
    plt.grid(True)
    plt.xlabel(c,fontsize=14)
    plt.ylabel('Log Alert Duration [D]',fontsize=14)
    plt.savefig('BivariateAnalysis/Numerical/trasf_durationVS{}'.format(c))
    plt.show()

### Target Variable VS Categorical Predictors - ReducedClasses
subset = data #data[data["closing_time"] < 60]
for c in cat_vars:
    x = subset['duration'].values
    y = subset[c].values
    print("####################################################################################################")
    print("\n\033[1m" + "Analisi Bivariata closing_time" + "-" + c + "\033[0;0m \n")
    data.boxplot(column="duration",by= c, figsize= (8,8), rot=45) 
    plt.savefig('BivariateAnalysis/ReducedClasses/trasf_durationVS{}_box.png'.format(c))
    plt.show()
    print("####################################################################################################")
