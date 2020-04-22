# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:44:02 2020

@author: andre
REGRESSION TREES
the DecisionTreeClassifier class for classification problems
the DecisionTreeRegressor class for regression.
In any case you need to one-hot encode categorical variables before you fit a tree with sklearn.
RANDOM FOREST REGRESSION
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('DataSets/encoded.csv')
data.columns

# convert duration data from str to timedelta64
#data['duration'] = pd.to_timedelta(data['duration'], unit='d')
#data.info()

# convert from timedelta64 to float64
#data['duration'] = data['duration'] / np.timedelta64(1, 'h')

predictors = data.iloc[:,:-1]
predictors.columns
n_features = predictors.shape[1]

# create directory
import os
if not os.path.exists('RegressionTree'):
    os.makedirs('RegressionTree')

#%% Regression Tree
# create a regressor object 
tree = DecisionTreeRegressor(max_depth=5, random_state = 0, max_leaf_nodes=35)
y = data['duration']
X = predictors

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit the regressor with X and Y data  from training set
tree.fit(X_train, y_train) 

# predict
y_pred = tree.predict(X_test)

# comparison
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

#%% Performances
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(tree.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(tree.score(X_test, y_test)))

#%% Feature Importance
print("Relative importance of the features: ",tree.feature_importances_)
with plt.style.context('dark_background'):
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.yticks(range(n_features+1,1,-1),predictors.columns,fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters",fontsize=15)
    plt.ylabel("Features\n",fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features+1,1,-1),width=tree.feature_importances_,height=0.5)
    plt.savefig('RegressionTree/ParametersImportance_tree.png')

#%% Visualizing Results
# import export_graphviz 
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(tree, out_file ='tree.dot') 
               #feature_names =['duration']
              
''' Single Regression Tree is highly overfitting data'''

#%% Random Forest Regressor

rfr = RandomForestRegressor(max_depth=50, random_state=0,max_features='auto',
                              max_leaf_nodes=50,n_estimators=100)

# fit the regressor with X and Y data  from training set
rfr.fit(X_train, y_train)

# predict
y_pred = rfr.predict(X_test)

# comparison
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

#%% Performances
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(rfr.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rfr.score(X_test, y_test)))

#%% Feature Importance
print("Relative importance of the features: ",rfr.feature_importances_)
with plt.style.context('dark_background'):
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.yticks(range(n_features+1,1,-1),predictors.columns,fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters",fontsize=15)
    plt.ylabel("Features\n",fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features+1,1,-1),width=rfr.feature_importances_,height=0.5)
    plt.savefig('RegressionTree/ParametersImportance_rfr.png')

    
#%% Comparison with Linear Model
''' Show the relative importance of regressors side by side
For Random Forest Model, show the relative importance of features as determined by 
the meta-estimator. For the OLS model, show normalized t-statistic values.

It will be clear that although the RandomForest regressor identifies the 
important regressors correctly, it does not assign the same level of relative 
importance to them as done by OLS method t-statistic.'''

#df_importance = pd.DataFrame(data=[rfr.feature_importances_,fitted.tvalues[1:]/sum(fitted.tvalues[1:])],
#                             columns=predictors.columns,
#                             index=['RF Regressor relative importance', 'OLS method normalized t-statistic'])
#df_importance


#%% Removing useless predictors
# rerun Random Forest Regressor
col = predictors.columns[rfr.feature_importances_ > 1e-3]
X = predictors[col]
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rfr = RandomForestRegressor(max_depth=50, random_state=0,max_features='auto',
                              max_leaf_nodes=50,n_estimators=100)

# fit the regressor with X and Y data  from training set
rfr.fit(X_train, y_train)

# predict
y_pred = rfr.predict(X_test)

# comparison
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(rfr.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rfr.score(X_test, y_test)))

#%% Analysis of Results
print("Relative importance of the features: ",rfr.feature_importances_)
with plt.style.context('dark_background'):
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.yticks(range(n_features+1,1,-1),X.columns,fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters",fontsize=15)
    plt.ylabel("Features\n",fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features+1,1,-1),width=rfr.feature_importances_,height=0.5)
    plt.savefig('RegressionTree/ParametersImportance_rfr2.png')
    
#%% Plots 

# fitted VS training set
fitted = rfr.predict(X_train)
plt.figure(figsize=(12,8))
plt.plot(y_train.reset_index(drop=True), label='y_train',marker='.')
plt.plot(fitted, label='fitted')
plt.xlabel("Training Set observations",fontsize=15)
plt.ylabel("LOG(Duration)",fontsize=15)
plt.title("Fitted vs. TrainSet",fontsize=18)
plt.legend()
plt.savefig('RegressionTree/fittedVStraining.png')
plt.show()

# predicted VS test set
df.reset_index(drop=True).plot(figsize=(12,8),marker='.')
plt.xlabel("Test Set observations",fontsize=15)
plt.ylabel("LOG(Duration)",fontsize=15)
plt.title("Predicted vs. TestSet",fontsize=18)
plt.savefig('RegressionTree/predictedVStest.png')
plt.show()

# predicted VS test set - TRASFORMED
df = np.exp(df)
df.reset_index(drop=True).plot(figsize=(12,8),marker='.')
plt.xlabel("Test Set observations",fontsize=15)
plt.ylabel("Duration",fontsize=15)
plt.title("Predicted vs. TestSet",fontsize=18)
plt.savefig('RegressionTree/predictedVStest_transf.png')
plt.show()


# Fitted VS Residuals
res = y_train - fitted
plt.figure(figsize=(12,8))
plt.scatter(x=fitted,y=res,edgecolor='k')
xmin=min(fitted)
xmax = max(fitted)
plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
plt.xlabel("Fitted values",fontsize=15)
plt.ylabel("Residuals",fontsize=15)
plt.title("Fitted vs. residuals plot",fontsize=18)
plt.grid(True)
plt.savefig('RegressionTree/fittedVSresiduals.png')
plt.show()

#%% Random Search Cross Validation - Tuning Random Forest Parameter
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Max number of final classes
max_leaf_nodes = [int(x) for x in np.linspace(10, 300, num = 30)]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'max_leaf_nodes' : max_leaf_nodes}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=None, 
                               n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

# View best params
rf_random.best_params_

# Comparison
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = rfr
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
