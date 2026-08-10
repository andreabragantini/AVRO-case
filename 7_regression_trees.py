# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:44:02 2020

@author: andre
REGRESSION TREES
DecisionTreeClassifier class for classification problems
DecisionTreeRegressor class for regression.
In any case you need to one-hot encode categorical variables before you fit a tree with sklearn.
RANDOM FOREST REGRESSION
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from avro_common import print_section

print_section('REGRESSION TREES')

data = pd.read_csv('data_sets/encoded.csv')
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
if not os.path.exists('regression_tree'):
    os.makedirs('regression_tree')

#%% Regression Tree
# create a regressor object
print_section('Decision tree regression')
tree = DecisionTreeRegressor(max_depth=5, random_state = 0, max_leaf_nodes=35)
y = data['duration']
X = predictors

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit the regressor with X and Y data  from training set
tree.fit(X_train, y_train)

# persist the fitted model so the predicting/comparison scripts can run standalone
dump(tree, 'regression_tree/tree.joblib')

# predict
y_pred = tree.predict(X_test)

# comparison
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

#%% Performances
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Error on the original day scale (the model is trained on log(minutes))
days_actual = np.exp(y_test) / 1440
days_pred = np.exp(y_pred) / 1440
print('MAE on day scale: {:.1f} days'.format(metrics.mean_absolute_error(days_actual, days_pred)))
print('Median AE on day scale: {:.1f} days'.format(np.median(np.abs(days_actual - days_pred))))

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(tree.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(tree.score(X_test, y_test)))

#%% Feature Importance
# (single tree importance is displayed but not saved; the RF version is saved below)
with plt.style.context('dark_background'):
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.yticks(range(n_features+1,1,-1),predictors.columns,fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters",fontsize=15)
    plt.ylabel("Features\n",fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features+1,1,-1),width=tree.feature_importances_,height=0.5)
    #plt.savefig('regression_tree/ParametersImportance_tree.png')  # single tree overfits; RF importance is the useful one

#%% Visualizing Results
# import export_graphviz
from sklearn.tree import export_graphviz

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
export_graphviz(tree, out_file ='regression_tree/tree.dot')
               #feature_names =['duration']

''' Single Regression Tree is highly overfitting data'''

#%% Random Forest Regressor
print_section('Random forest regression')

rfr = RandomForestRegressor(max_depth=50, random_state=0,max_features='sqrt',
                              max_leaf_nodes=50,n_estimators=100)

# fit the regressor with X and Y data  from training set
rfr.fit(X_train, y_train)

# persist the fitted model so the predicting/comparison scripts can run standalone
dump(rfr, 'regression_tree/rfr.joblib')

# predict
y_pred = rfr.predict(X_test)

# comparison
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

#%% Performances
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Error on the original day scale (the model is trained on log(minutes))
days_actual = np.exp(y_test) / 1440
days_pred = np.exp(y_pred) / 1440
print('MAE on day scale: {:.1f} days'.format(metrics.mean_absolute_error(days_actual, days_pred)))
print('Median AE on day scale: {:.1f} days'.format(np.median(np.abs(days_actual - days_pred))))

# compute and print the R Square
print('R-squared score (training): {:.3f}'.format(rfr.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'.format(rfr.score(X_test, y_test)))

# Comparable metrics table (shared format across model directories)
from avro_common import compute_model_metrics, write_metrics_table
tree_metrics = compute_model_metrics(
    np.exp(y_test) / 1440, np.exp(tree.predict(X_test)) / 1440,
    r2_log=tree.score(X_test, y_test),
)
rfr_metrics = compute_model_metrics(
    days_actual, days_pred, r2_log=rfr.score(X_test, y_test)
)
write_metrics_table(
    'regression_tree/model_metrics.txt',
    [('Decision tree', tree_metrics), ('Random forest', rfr_metrics)],
    title='Regression trees - model metrics (test set, day scale)',
)

#%% Feature Importance
# (feature importances are shown in the saved bar chart: regression_tree/ParametersImportance_rfr.png)
with plt.style.context('dark_background'):
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.yticks(range(n_features+1,1,-1),predictors.columns,fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters",fontsize=15)
    plt.ylabel("Features\n",fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features+1,1,-1),width=rfr.feature_importances_,height=0.5)
    plt.savefig('regression_tree/ParametersImportance_rfr.png')


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


#%% Analysis of Results
# (RandomForest importance was saved above in ParametersImportance_rfr.png)
with plt.style.context('dark_background'):
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.yticks(range(n_features+1,1,-1),X.columns,fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters",fontsize=15)
    plt.ylabel("Features\n",fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features+1,1,-1),width=rfr.feature_importances_,height=0.5)
    #plt.savefig('regression_tree/ParametersImportance_rfr2.png')  # duplicate of ParametersImportance_rfr.png

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
#plt.savefig('regression_tree/fittedVStraining.png')  # training fit; test fit is what matters
plt.close()

# predicted VS test set (same chart as multi_lin_reg/predictedVStest.png)
df_test = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),
    'RegrTree': tree.predict(X_test),
    'RandomForest': rfr.predict(X_test),
})
df_test.plot(figsize=(12,8),marker='.')
plt.xlabel("Test Set observations",fontsize=15)
plt.ylabel("LOG(Duration)",fontsize=15)
plt.title("Predicted vs. TestSet - regression trees (log scale)",fontsize=16)
plt.grid(True)
plt.savefig('regression_tree/predictedVStest.png')
plt.close()

# predicted VS test set - TRASFORMED (original day scale, same as multi_lin_reg)
df_test_exp = np.exp(df_test) / 1440.0
df_test_exp.plot(figsize=(12,8),marker='.')
plt.xlabel("Test Set observations",fontsize=15)
plt.ylabel("Duration (days)",fontsize=15)
plt.title("Predicted vs. TestSet - regression trees (day scale)",fontsize=16)
plt.grid(True)
plt.savefig('regression_tree/predictedVStest_transf.png')
plt.close()


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
#plt.savefig('regression_tree/fittedVSresiduals.png')  # duplicates multi_lin_reg/resVSfit.png
plt.close()

#%% Random Search Cross Validation - Tuning Random Forest Parameter
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 100)]
# Number of features to consider at every split
max_features = ['sqrt', None]
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
print_section('Hyperparameter tuning (RandomizedSearchCV)')
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 25, cv = 3, verbose=0, random_state=42,
                               n_jobs = 1)
# Fit the random search model
rf_random.fit(X_train, y_train)

# View best params
print('Best hyperparameters:', rf_random.best_params_)

# Comparison
def evaluate(model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    errors = abs(predictions - y_eval)
    mape = 100 * np.mean(errors / y_eval)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} days.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

base_model = rfr
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
