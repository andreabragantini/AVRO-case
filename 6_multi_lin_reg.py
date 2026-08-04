# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:01:47 2020

@author: andre
MODEL DESIGN - MULTIPLE LINEAR REGRESSION
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from avro_common import print_section

print_section('MODEL DESIGN - MULTIPLE LINEAR REGRESSION')


data = pd.read_csv('data_sets/encoded.csv')
data.columns

# convert duration data from str to timedelta64
#data['duration'] = pd.to_timedelta(data['duration'], unit='d')
#data.info()

# convert from timedelta64 to float64
#data['duration'] = data['duration'] / np.timedelta64(1, 'h')

predictors = data.iloc[:,:-1]
predictors.columns

# create directory
import os
if not os.path.exists('multi_lin_reg'):
    os.makedirs('multi_lin_reg')

#%% Results from features selection
if 'b1' in globals():
    col = list(predictors.columns[b1])
else:
    col = ['vote_count', 'comment_count', 'watch_count']

predictors[col].describe()
      
#%% Multiple Linear Regression - Sklearn
# Use the forward feature-selection output when available.
print_section('Multiple Linear Regression (sklearn)')
selected = predictors[col]

X=selected
y=data['duration']

# In case of polynomial regression:
#polynomial_features= PolynomialFeatures(degree=2)
#X = polynomial_features.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state = 0)
# Fit a multivariate regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))

# predict
y_pred = linreg.predict(X_test)

# comparison
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Error on the original day scale (the model is trained on log(minutes))
days_actual = np.exp(y_test) / 1440
days_pred = np.exp(y_pred) / 1440
print('MAE on day scale: {:.1f} days'.format(metrics.mean_absolute_error(days_actual, days_pred)))
print('Median AE on day scale: {:.1f} days'.format(np.median(np.abs(days_actual - days_pred))))

#%% Multiple Linear Regression - statsmodel.OLS()
import statsmodels.formula.api as sm

print_section('Multiple Linear Regression (statsmodels OLS)')

# Creating a formula string for using in the statsmodels.OLS()
formula_str = data.columns[-1]+' ~ '+'+'.join(col)                             # selected feature model
#formula_str = data.columns[-1]+' ~ '+'+'.join(predictors.columns)                            # full feature model

# Construct and fit the model. Print summary of the fitted model
model=sm.ols(formula=formula_str, data=pd.concat([X_train,y_train],axis=1))
fitted = model.fit()
print(fitted.summary())

# get estimated parameters
coef = pd.DataFrame(fitted.params, columns=['value'])

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(fitted.rsquared))

# A new Result dataframe: p-values and statistical significance of the features
df_result=pd.DataFrame()
df_result['pvalues']=fitted.pvalues[1:]
df_result['Features']= col
#df_result['Features']= predictors.columns
df_result.set_index('Features',inplace=True)
def yes_no(b):
    if b < 0.05:
        return 'Yes'
    else:
        return 'No'

df_result['Statistically significant?']= df_result['pvalues'].apply(yes_no)
df_result

#%% Analysis of Residuals
print_section('Residual diagnostics')

# Residuals vs. predicting variables plots
for c in col:
    # One plot per one-hot reporter column is overwhelming; save only the
    # meaningful non-reporter predictors.
    if c.startswith('reporter'):
        continue
    plt.figure(figsize=(12,8))
    plt.title("{} vs. Model residuals".format(c),fontsize=16)
    plt.scatter(x=X_train[c],y=fitted.resid,color='blue',edgecolor='k')
    plt.grid(True)
    xmin=min(data[c])
    xmax = max(data[c])
    plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
    plt.xlabel(c,fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.savefig(f'multi_lin_reg/residualsVS{c}.png')
    plt.close()

# Fitted VS Residuals
plt.figure(figsize=(12,8))
p=plt.scatter(x=fitted.fittedvalues,y=fitted.resid,edgecolor='k')
xmin=min(fitted.fittedvalues)
xmax = max(fitted.fittedvalues)
plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
plt.xlabel("Fitted values",fontsize=15)
plt.ylabel("Residuals",fontsize=15)
plt.title("Fitted vs. residuals plot",fontsize=18)
plt.grid(True)
plt.savefig('multi_lin_reg/resVSfit.png')
plt.close()
    
# Histogram of normalized residuals
plt.figure(figsize=(12,8))
plt.hist(fitted.resid_pearson,bins=20,edgecolor='k')
plt.ylabel('Count',fontsize=15)
plt.xlabel('Normalized residuals',fontsize=15)
plt.title("Histogram of normalized residuals",fontsize=18)
plt.savefig('multi_lin_reg/resHist.png')
plt.close()

# QQ plot of residuals
from statsmodels.graphics.gofplots import qqplot

plt.figure(figsize=(12,8))
fig=qqplot(fitted.resid_pearson,line='45',fit='True')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Theoretical quantiles",fontsize=15)
plt.ylabel("Sample quantiles",fontsize=15)
plt.title("Q-Q plot of normalized residuals",fontsize=18)
plt.grid(True)
plt.savefig('multi_lin_reg/resQQplot.png')
plt.close()

# Shapiro Test on Residuals
from scipy.stats import shapiro

_,p=shapiro(fitted.resid)
if p<0.01:
    print("The residuals seem to come from Gaussian process")
else:
    print("The normality assumption may not hold")

#%% Prediction
print_section('Prediction on test set')
y_pred = fitted.predict(X_test)

# comparison and performances
df_ols=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

print('R-squared score (test): {:.3f}'.format(metrics.r2_score(y_test, y_pred)))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Error on the original day scale (the model is trained on log(minutes))
days_actual = np.exp(y_test) / 1440
days_pred = np.exp(y_pred) / 1440
print('MAE on day scale: {:.1f} days'.format(metrics.mean_absolute_error(days_actual, days_pred)))
print('Median AE on day scale: {:.1f} days'.format(np.median(np.abs(days_actual - days_pred))))

#%% Plots

# predicted VS test set
df_ols.reset_index(drop=True).plot(figsize=(12,8),marker='.')
plt.xlabel("Test Set observations",fontsize=15)
plt.ylabel("LOG(Duration)",fontsize=15)
plt.title("Predicted vs. TestSet",fontsize=18)
plt.savefig('multi_lin_reg/predictedVStest.png')
plt.close()

# predicted VS test set - TRASFORMED
df_ols = np.exp(df_ols)
df_ols.reset_index(drop=True).plot(figsize=(12,8),marker='.')
plt.xlabel("Test Set observations",fontsize=15)
plt.ylabel("Duration",fontsize=15)
plt.title("Predicted vs. TestSet",fontsize=18)
plt.savefig('multi_lin_reg/predictedVStest_transf.png')
plt.close()






