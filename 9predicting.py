# -*- coding: utf-8 -*-
"""
Train the final models and extract predictions for the validation set.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from avro_common import SELECTED_FEATURES, prepare_validation_features

if not os.path.exists("Question2"):
    os.makedirs("Question2")

training = pd.read_csv("DataSets/encoded.csv")
validation_raw = pd.read_csv("DataSets/validationset.csv")

X_train = training[SELECTED_FEATURES]
y_train = training["duration"]
test = prepare_validation_features(validation_raw, X_train.columns.tolist())

linreg = LinearRegression()
linreg.fit(X_train, y_train)

tree = DecisionTreeRegressor(max_depth=5, random_state=0, max_leaf_nodes=35)
tree.fit(X_train, y_train)

rfr = RandomForestRegressor(
    max_depth=50,
    random_state=0,
    max_features="sqrt",
    max_leaf_nodes=50,
    n_estimators=100,
)
rfr.fit(X_train, y_train)

lr_pred = linreg.predict(test)
tree_pred = tree.predict(test)
rf_pred = rfr.predict(test)

df_pred = pd.DataFrame(
    {"LinearModel": lr_pred, "RegrTree": tree_pred, "RandomForest": rf_pred},
    index=test.index,
)

df_pred.plot(figsize=(12, 5), marker=".")
plt.xlabel("Validation Set observations", fontsize=15)
plt.ylabel("LOG(Duration)", fontsize=15)
plt.title("Validation Set Prediction - transformed", fontsize=18)
plt.savefig("Question2/predictionComparison_log.png")
plt.show()

df_pred_minutes = np.exp(df_pred)
df_pred_minutes.plot(figsize=(12, 5), marker=".")
plt.xlabel("Validation Set observations", fontsize=15)
plt.ylabel("Duration", fontsize=15)
plt.title("Validation Set Prediction", fontsize=18)
plt.savefig("Question2/predictionComparison.png")
plt.show()

df_pred_minutes["LinearModel"] = pd.to_timedelta(df_pred_minutes["LinearModel"], unit="m")
df_pred_minutes["RegrTree"] = pd.to_timedelta(df_pred_minutes["RegrTree"], unit="m")
df_pred_minutes["RandomForest"] = pd.to_timedelta(df_pred_minutes["RandomForest"], unit="m")
print(df_pred_minutes)

#%% 3 interesting cases
df_pred_case = df_pred_minutes.copy()

for idx in [321, 120, 235]:
    print("\nCASE", idx)
    print(validation_raw.loc[idx, ["status", "priority", "issue_type", "reporter"]])
    print(df_pred_case.loc[idx])

