# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:39:20 2020

@author: andre
RIDGE REGRESSION
LASSO REGRESSION
"""

#%% RIDGE REGRESSION
# Least Angle Regression model a.k.a. LARS package

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)

# Scale the X_train and X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a ridge regression with alpha=20
linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

# Print the training R squared
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
# Print the test Rsquared
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

trainingRsquared=[]
testRsquared=[]
# Plot the effect of alpha on the test Rsquared
print('Ridge regression: effect of alpha regularization parameter\n')
# Choose a list of alpha values
for this_alpha in [0.001,.01,.1,0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    # Compute training rsquared
    r2_train = linridge.score(X_train_scaled, y_train)
    # Compute test rsqaured
    r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    trainingRsquared.append(r2_train)
    testRsquared.append(r2_test)

# Create a dataframe
alpha=[0.001,.01,.1,0, 1, 10, 20, 50, 100, 1000]    
trainingRsquared=pd.DataFrame(trainingRsquared,index=alpha)
testRsquared=pd.DataFrame(testRsquared,index=alpha)

# Plot training and test R squared as a function of alpha
df3=pd.concat([trainingRsquared,testRsquared],axis=1)
df3.columns=['trainingRsquared','testRsquared']

'''The plot below shows the training and test error when increasing the tuning 
or regularization parameter ‘alpha’'''

fig5=df3.plot(figsize=(12,8), marker='.')
fig5=plt.title('Ridge training and test squared error vs Alpha')
fig5=plt.xlabel('alpha')
fig5=plt.ylabel('SE')
fig5.figure.savefig('FeatureSelection/fig5.png', bbox_inches='tight')


### Plot the coefficient shrinkage using the LARS package
from sklearn import linear_model
###############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(0, 8, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

###############################################################################
# Display results

ax = plt.gca()

fig6=ax.figure.set_size_inches(12,8)
fig6=ax.plot(alphas, coefs)
fig6=ax.set_xscale('log')
fig6=ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
fig6=plt.xlabel('alpha')
fig6=plt.ylabel('weights')
fig6=plt.title('Ridge coefficients as a function of the regularization')
fig6=plt.axis('tight')
plt.savefig('FeatureSelection/fig6.png', bbox_inches='tight')

'''For Python the coefficient shrinkage with LARS must be viewed from right 
to left, where you have increasing alpha. 
As alpha increases the coefficients shrink to 0.'''

#%% LASSO REGULARIZATION
'''The Lasso is another form of regularization, also known as L1 regularization. 
Unlike the Ridge Regression where the coefficients of features which do not 
influence the target tend to zero, in the lasso regualrization the coefficients become 0.'''

from sklearn.linear_model import Lasso

linlasso = Lasso(alpha=1000, max_iter = 10000).fit(X_train_scaled, y_train)

print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))

print('Features with non-zero weight (sorted by absolute magnitude):')
for e in sorted(list(zip(list(x), linlasso.coef_)), key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))
        

print('Lasso regression: effect of alpha regularization\n\
parameter on number of features kept in final model\n')

trainingRsquared=[]
testRsquared=[]
#for alpha in [0.01,0.05,0.1, 1, 2, 3, 5, 10, 20, 50]:
for alpha in [0.01,0.07,0.05, 0.1, 1,2, 3, 5, 10, 100, 1000, 10000]:
    linlasso = Lasso(alpha, max_iter = 1000000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    trainingRsquared.append(r2_train)
    testRsquared.append(r2_test)
    
alpha=[0.01,0.07,0.05, 0.1, 1,2, 3, 5, 10, 100, 1000, 10000]    
#alpha=[0.01,0.05,0.1, 1, 2, 3, 5, 10, 20, 50]
trainingRsquared=pd.DataFrame(trainingRsquared,index=alpha)
testRsquared=pd.DataFrame(testRsquared,index=alpha)

df3=pd.concat([trainingRsquared,testRsquared],axis=1)
df3.columns=['trainingRsquared','testRsquared']

fig7=df3.plot(figsize=(12,8), marker='.')
fig7=plt.title('LASSO training and test squared error vs Alpha')
fig5=plt.xlabel('alpha')
fig5=plt.ylabel('SE')
fig7.figure.savefig('FeatureSelection/fig7.png', bbox_inches='tight')

### Plot the coefficient shrinkage using the LARS package
print("Computing regularization path using the LARS ...")
alphas, _, coefs = linear_model.lars_path(X_train_scaled, y_train, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

fig8=plt.figure(figsize=(12,8))
fig8=plt.plot(xx, coefs.T, marker='.')

ymin, ymax = plt.ylim()
fig8=plt.vlines(xx, ymin, ymax, linestyle='dashed')
fig8=plt.xlabel('|coef| / max|coef|')
fig8=plt.ylabel('Coefficients')
fig8=plt.title('LASSO Path - Coefficient Shrinkage vs L1')
fig8=plt.axis('tight')
plt.savefig('FeatureSelection/fig8.png', bbox_inches='tight')















