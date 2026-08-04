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
import re
from avro_common import print_section

print_section('3. BIVARIATE ANALYSIS')

# load data
data = pd.read_csv('data_sets/processed.csv')
data.columns = data.columns.map(lambda col: str(col).strip().replace('\ufeff', ''))
data.duration = pd.to_timedelta(data['duration']).dt.total_seconds() / 60
data.columns
cat_vars = [ col for col in data.columns if data[col].dtype.kind not in 'biufcm']
cat_vars.remove('reporter')
num_vars = [ col for col in data.columns if data[col].dtype.kind in 'biufc' ]
num_vars.remove('duration')
target = data.duration

# NB The meaning of biufc: b bool, i int (signed), u unsigned int, f float, c complex, m timedelta

# create directories
import os


def safe_name(value):
    """Return a filesystem-safe token for plot file names."""
    value = str(value).strip().replace('\ufeff', '')
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', value)
    value = re.sub(r'\s+', '_', value)
    return value or 'unnamed'


if not os.path.exists('bivariate_analysis/numerical'):
    os.makedirs('bivariate_analysis/numerical')
if not os.path.exists('bivariate_analysis/full_classes'):
    os.makedirs('bivariate_analysis/full_classes')
if not os.path.exists('bivariate_analysis/reduced_classes'):
    os.makedirs('bivariate_analysis/reduced_classes')
if not os.path.exists('exploratory_analysis'):
    os.makedirs('exploratory_analysis')

print_section('Target vs categorical predictors (full classes)')

#%% Target Variable VS Categorical Predictors - FullClasses

subset = data #data[data["closing_time"] < 60]
for c in cat_vars:
    c_safe = safe_name(c)
    x = subset['duration'].values
    y = subset[c].values
        
    confusion_matrix = subset.groupby(['duration', c]).size().sort_values(ascending = False).unstack(fill_value=0)
    confusion_matrix = confusion_matrix.reindex(confusion_matrix.sum().sort_values(ascending = False).index, axis=1)
    
    confusion_matrix = confusion_matrix / confusion_matrix.sum()

    confusion_matrix.plot.line(title = "Analisi Bivariata duration" + "-" + c, figsize= (8,8))
    plt.savefig('bivariate_analysis/full_classes/durationVS{}.png'.format(c_safe))
    data.boxplot(column="duration",by= c, figsize= (8,8)) 
    plt.savefig('bivariate_analysis/full_classes/durationVS{}_box.png'.format(c_safe))
    plt.close()

print_section('Reduce classes: reporters and issue types')

#%% Reduce Classes - Combine Methods
''' The objective is to try to reduce the many levels present in our categorical vars'''

# Reporter
data['reporter'].value_counts()[:40].plot(kind='bar')
plt.title('Most frequent Reporters - Training Set')
plt.tight_layout()
plt.savefig('exploratory_analysis/Reporters.png', bbox_inches='tight')
plt.close()
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
plt.savefig('exploratory_analysis/Issue_types.png', bbox_inches='tight')
plt.close()
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
#plt.savefig('exploratory_analysis/Priorities.png', bbox_inches='tight')
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
    
print_section('Target vs categorical predictors (reduced classes)')

#%% Target Variable VS Categorical Predictors - ReducedClasses
cat_vars.append('reporter')

subset = data #data[data["closing_time"] < 60]
for c in cat_vars:
    c_safe = safe_name(c)
    x = subset['duration'].values
    y = subset[c].values
        
    confusion_matrix = subset.groupby(['duration', c]).size().sort_values(ascending = False).unstack(fill_value=0)
    confusion_matrix = confusion_matrix.reindex(confusion_matrix.sum().sort_values(ascending = False).index, axis=1)
    
    confusion_matrix = confusion_matrix / confusion_matrix.sum()

    confusion_matrix.plot.line(title = "Analisi Bivariata duration" + "-" + c, figsize= (8,8))
    plt.savefig('bivariate_analysis/reduced_classes/durationVS{}.png'.format(c_safe))
    data.boxplot(column="duration",by= c, figsize= (8,8), rot=45) 
    plt.savefig('bivariate_analysis/reduced_classes/durationVS{}_box.png'.format(c_safe))
    plt.close()

print_section('Target vs numerical predictors')

#%% Target Variable VS Numerical Predictors

# simple scatter plots 
for c in num_vars:
    c_safe = safe_name(c)
    plt.figure(figsize=(12,8))
    plt.title("{} vs. duration".format(c),fontsize=16)
    plt.scatter(x=data[c],y=target,color='blue',edgecolor='k')
    plt.grid(True)
    plt.xlabel(c,fontsize=14)
    plt.ylabel('Alert Duration [D]',fontsize=14)
    plt.savefig('bivariate_analysis/numerical/durationVS{}.png'.format(c_safe))
    plt.close()

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
#    plt.savefig('bivariate_analysis/numerical/durationVS{}_Hex'.format(num_vars[i]))
#    # Show the plot
#    plt.close()
    
#%% Pairplots
from seaborn import pairplot

num_vars.append('duration')
pairplot(data[num_vars])    
plt.title('Pairplot for numerical features')
plt.savefig('bivariate_analysis/numerical/pairplot_num_vars.png')
plt.close()

#%% Multi-Collinearity check between numerical predictors

from statsmodels.graphics.correlation import plot_corr

num_vars.remove('duration')
corr = data[num_vars].corr()
corr

fig = plot_corr(corr,xnames=corr.columns)
plt.savefig('bivariate_analysis/numerical/heatmap.png')
plt.close()

''' The heatmap shows some correlation between 'watch_count' and both
'vote_count' and 'comment_count'. These last two are also a bit correlated.'''

print_section('Log transform of skewed features')

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
plt.savefig('bivariate_analysis/numerical/pairplot_num_vars_log.png')
plt.close()

# trasformed duration (target variable)
#plt.figure(figsize=(12,8))
data['duration'].hist()
plt.ylabel('N# of observations')
plt.xlabel('Log(Time)')
plt.title('Log-trasformed target variable')
plt.savefig('exploratory_analysis/duration_log.png', bbox_inches='tight')
plt.close()


#%% Save trasformed dataset
data.to_csv('data_sets/trasformed_nonencoded.csv', index=False)

print_section('Bivariate analysis on log-transformed data')

#%% Repeat Bivariate Analysis on trasformed data

### Target Variable VS Numerical Predictors
# simple scatter plots 
for c in num_vars:
    c_safe = safe_name(c)
    #plt.figure(figsize=(12,8))
    plt.title("{} vs. duration (transformed)".format(c),fontsize=16)
    plt.scatter(x=data[c],y=data.duration,color='blue',edgecolor='k')
    plt.grid(True)
    plt.xlabel(c,fontsize=14)
    plt.ylabel('Log Alert Duration [D]',fontsize=14)
    plt.savefig('bivariate_analysis/numerical/trasf_durationVS{}.png'.format(c_safe))
    plt.close()

### Target Variable VS Categorical Predictors - ReducedClasses
subset = data #data[data["closing_time"] < 60]
for c in cat_vars:
    c_safe = safe_name(c)
    x = subset['duration'].values
    y = subset[c].values
    data.boxplot(column="duration",by= c, figsize= (8,8), rot=45) 
    plt.savefig('bivariate_analysis/reduced_classes/trasf_durationVS{}_box.png'.format(c_safe))
    plt.close()
