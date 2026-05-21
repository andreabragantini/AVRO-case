# -*- coding: utf-8 -*-
"""
Regression tree and random forest models.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from avro_common import SELECTED_FEATURES

data = pd.read_csv("DataSets/encoded.csv")
predictors = data[SELECTED_FEATURES]
n_features = predictors.shape[1]

if not os.path.exists("RegressionTree"):
    os.makedirs("RegressionTree")

X = predictors
y = data["duration"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

tree = DecisionTreeRegressor(max_depth=5, random_state=0, max_leaf_nodes=35)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Tree MAE:", np.mean(np.abs(y_test - y_pred)))
print("Tree MSE:", np.mean((y_test - y_pred) ** 2))
print("Tree RMSE:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("R-squared score (training): {:.3f}".format(tree.score(X_train, y_train)))
print("R-squared score (test): {:.3f}".format(tree.score(X_test, y_test)))

with plt.style.context("dark_background"):
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.yticks(range(n_features + 1, 1, -1), predictors.columns, fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters", fontsize=15)
    plt.ylabel("Features\n", fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features + 1, 1, -1), width=tree.feature_importances_, height=0.5)
    plt.savefig("RegressionTree/ParametersImportance_tree.png")

export_graphviz(tree, out_file="RegressionTree/tree.dot")

print("Single regression tree is highly overfitting data")

rfr = RandomForestRegressor(
    max_depth=50,
    random_state=0,
    max_features="sqrt",
    max_leaf_nodes=50,
    n_estimators=100,
)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

print("RF MAE:", np.mean(np.abs(y_test - y_pred)))
print("RF MSE:", np.mean((y_test - y_pred) ** 2))
print("RF RMSE:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("R-squared score (training): {:.3f}".format(rfr.score(X_train, y_train)))
print("R-squared score (test): {:.3f}".format(rfr.score(X_test, y_test)))

with plt.style.context("dark_background"):
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.yticks(range(n_features + 1, 1, -1), predictors.columns, fontsize=20)
    plt.xlabel("Relative (normalized) importance of parameters", fontsize=15)
    plt.ylabel("Features\n", fontsize=20)
    plt.tight_layout()
    plt.barh(range(n_features + 1, 1, -1), width=rfr.feature_importances_, height=0.5)
    plt.savefig("RegressionTree/ParametersImportance_rfr.png")

fitted = rfr.predict(X_train)
plt.figure(figsize=(12, 8))
plt.plot(y_train.reset_index(drop=True), label="y_train", marker=".")
plt.plot(fitted, label="fitted")
plt.xlabel("Training Set observations", fontsize=15)
plt.ylabel("LOG(Duration)", fontsize=15)
plt.title("Fitted vs. TrainSet", fontsize=18)
plt.legend()
plt.savefig("RegressionTree/fittedVStraining.png")
plt.show()

df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
df.reset_index(drop=True).plot(figsize=(12, 8), marker=".")
plt.xlabel("Test Set observations", fontsize=15)
plt.ylabel("LOG(Duration)", fontsize=15)
plt.title("Predicted vs. TestSet", fontsize=18)
plt.savefig("RegressionTree/predictedVStest.png")
plt.show()

df = np.exp(df)
df.reset_index(drop=True).plot(figsize=(12, 8), marker=".")
plt.xlabel("Test Set observations", fontsize=15)
plt.ylabel("Duration", fontsize=15)
plt.title("Predicted vs. TestSet", fontsize=18)
plt.savefig("RegressionTree/predictedVStest_transf.png")
plt.show()

res = y_train - fitted
plt.figure(figsize=(12, 8))
plt.scatter(x=fitted, y=res, edgecolor="k")
plt.hlines(y=0, xmin=min(fitted) * 0.9, xmax=max(fitted) * 1.1, color="red", linestyle="--", lw=3)
plt.xlabel("Fitted values", fontsize=15)
plt.ylabel("Residuals", fontsize=15)
plt.title("Fitted vs. residuals plot", fontsize=18)
plt.grid(True)
plt.savefig("RegressionTree/fittedVSresiduals.png")
plt.show()

