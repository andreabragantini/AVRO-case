# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:03:50 2020

@author: andre
TASK: develop a data science model with the data in the CSV file to predict the required
time to resolve an alert, based on its characteristics.

Script used for exploratory analysis of the given dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
import seaborn as sns
sns.set()

# read data
dataset = pd.read_csv('avro-issues.csv')
dataset.head()
dataset.columns
n = len(dataset)

# Directory creation
import os
if not os.path.exists('ExploratoryAnalysis'):
    os.mkdir('ExploratoryAnalysis')

#%% Exploration - Categorical Variables

# var "status"
col = "status"
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count), col))
print(count)

plt.figure(figsize=(12,8))
dataset[col].value_counts().plot(kind='bar')
plt.title('Status - Full Dataset')
plt.savefig('ExploratoryAnalysis/freq_status.png')

# var "priority"
col = 'priority'
count = dataset[col].value_counts()
print('\nFrequency of the %d categories for var %s'%(len(count), col))
print(count)

plt.figure(figsize=(12,8))
dataset[col].value_counts().plot(kind='bar')
plt.title('Priority - Full Dataset')
plt.savefig('ExploratoryAnalysis/freq_priority.png')

# var 'issue_type'
col = 'issue_type'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
dataset[col].value_counts().plot(kind='bar')
plt.title('Issue_type - Full Dataset')
plt.savefig('ExploratoryAnalysis/freq_issue_type.png')

# var 'resolution'
col = 'resolution'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
dataset[col].value_counts().plot(kind='bar')
plt.title('Resolution - Full Dataset')
plt.savefig('ExploratoryAnalysis/freq_resolution.png')

#%% Exploration - Numerical Variables

# var 'vote_count'
col = 'vote_count'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
dataset[col].value_counts().sort_index().plot(kind='bar')
plt.title('vote_count - Full Dataset')
plt.xlabel('N# of votes')
plt.ylabel('N# of alerts')
plt.savefig('ExploratoryAnalysis/freq_vote_count.png')

plt.figure(figsize=(12,8))
dataset[col].value_counts().sort_index().map(lambda x: np.log(x)).plot(kind='bar')
plt.title('vote_count - Full Dataset')
plt.xlabel('N# of votes')
plt.ylabel('LOG(N) number of alerts')
plt.savefig('ExploratoryAnalysis/freq_vote_count_log.png')

# var 'comment_count'
col = 'comment_count'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
count.sort_index().plot(kind='bar')
count.sort_index().plot.area()
plt.title('comment_count - Full Dataset')
plt.xlabel('N# of comments')
plt.ylabel('N# of alerts')
plt.savefig('ExploratoryAnalysis/freq_comm_count.png')

# var 'description_length'
col = 'description_length'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
count.sort_index().plot(marker='.')
plt.title('description_length - Full Dataset')
plt.xlabel('Length in characters')
plt.ylabel('N# of alerts')
plt.savefig('ExploratoryAnalysis/freq_descr_length.png')

''' There are a few outliers with very long descriptions'''
# Looking only at short descriptions
plt.figure(figsize=(12,8))
dataset[col].hist(bins=range(0,2000,100))
plt.title('Short description lengths')
plt.xlabel('Length in characters - class = 100 char')
plt.ylabel('N# of alerts')
plt.savefig('ExploratoryAnalysis/freq_descr_length_short.png')

# var 'summary_length'
col = 'summary_length'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
#count.sort_index().plot(kind='bar')
dataset[col].hist(bins=range(0,150,5))
plt.title('summary_length - Full Dataset')
plt.xlabel('Length in characters')
plt.ylabel('N# of alerts')
plt.savefig('ExploratoryAnalysis/freq_sum_length.png')

''' Possible Normal Distribution? '''
# q-q plot
qqplot(dataset[col], line='s')
plt.title('Q-Q plot: summary_length')
plt.show()
# normality test
stat, p = shapiro(dataset[col])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
        
# var 'watch_count'
col = 'watch_count'
count = dataset[col].value_counts()
print('\nFrequency of %d categories for var %s'%(len(count),col))
print(count)

plt.figure(figsize=(12,8))
count.sort_index().plot(kind='bar')
plt.title('watch_count - Full Dataset')
plt.xlabel('N# of watch')
plt.ylabel('N# of alerts')
plt.savefig('ExploratoryAnalysis/freq_watch_count.png')

''' After 7 watch they can be considered outliers'''

#%%######################################
#  Which features we should focus on ? 
#########################################

# closed alerts
closed = dataset.query('status == "Closed"')
len(closed)
closed['resolution']
set(closed['resolution'])               # closing casues 

closednotassigned = closed[pd.isna(closed['assignee'])]     # not assigned closed alerts
len(closednotassigned)

closednotassigned.query('resolution == "Fixed"')            # fixed but not assigned

# resolved alerts
resolved = dataset.query('status == "Resolved"')
len(resolved)
resolved['resolution']
set(resolved['resolution'])               # closing casues 

# different types of closing causes ['resolution'] could be roughly decreased to two principal ones:
fixed = dataset[dataset.resolution == 'Fixed']
len(fixed)

len(resolved) + len(closed)

closed[closed.resolution != 'Fixed']
resolved[resolved.resolution != 'Fixed']

#%% Plots
print('\nPlotting differences in dataset composition:')

fig, ax = plt.subplots(1,3, figsize=(15,10))
labels = ['Fixed','AllOthers-closed','AllOthers-resolved']
sizes = [len(fixed), 
         len(closed[closed.resolution != 'Fixed']),
         len(resolved[resolved.resolution != 'Fixed'])]
ax[0].pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax[0].legend(loc='lower left')
ax[0].set_title('Considered "resolution": tot = %s' % sum(sizes))

ax[1].bar(0,len(fixed), width=0.1)
ax[1].bar(0,len(closed[closed.resolution != 'Fixed']), width=0.1, bottom = len(fixed))
ax[1].bar(0,len(resolved[resolved.resolution != 'Fixed']), width=0.1, bottom = len(fixed)+len(closed[closed.resolution != 'Fixed']))
ax[1].bar(1,len(closed), width=0.1)
ax[1].bar(1,len(resolved), width=0.1, bottom = len(closed))
ax[1].set_xticks([0,1])
ax[1].set_yticks([len(closed),sum(sizes)])
ax[1].set_title('Comparison resolution vs status')

labels2 = ['Closed','Resolved']
sizes2 = [len(closed), 
         len(resolved)]
ax[2].set_title('Considered "status": tot = %s' % sum(sizes2))
ax[2].pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['red','purple'])
ax[2].legend(loc='lower left')

plt.tight_layout()
plt.savefig('ExploratoryAnalysis/comparison.png', bbox_inches='tight')
plt.show()

