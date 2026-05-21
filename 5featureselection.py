# -*- coding: utf-8 -*-
"""
Feature selection on the encoded dataset.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

data = pd.read_csv("DataSets/encoded.csv")
predictors = data.iloc[:, :-1]
y = data["duration"]
x = data.iloc[:, :-1]
lr = LinearRegression()

if not os.path.exists("FeatureSelection"):
    os.makedirs("FeatureSelection")


def best_subset(metric_dict):
    keys = sorted(metric_dict.keys())
    mse_by_key = []
    for key in keys:
        mse_by_key.append(-np.mean(metric_dict[key]["cv_scores"]))
    best_key = keys[int(np.argmin(mse_by_key))]
    return best_key, metric_dict[best_key]["feature_idx"], mse_by_key


def fit_search(forward, floating, label):
    search = SFS(
        lr,
        k_features=(1, predictors.shape[1]),
        forward=forward,
        floating=floating,
        scoring="neg_mean_squared_error",
        cv=5,
    )
    search = search.fit(x.values, y.values)
    metric_dict = search.get_metric_dict()
    best_key, best_idx, mse = best_subset(metric_dict)
    features = list(x.columns[list(best_idx)])
    print(f"{label}: best k = {best_key}")
    print(f"{label}: features = {features}")
    keys = sorted(metric_dict.keys())
    return keys, mse


keys, n1 = fit_search(True, False, "SFS_f")
_, n2 = fit_search(False, False, "SFS_b")
_, n3 = fit_search(True, True, "SFFS")
_, n4 = fit_search(False, True, "SFBS")

plt.plot(keys, n1, label="SFS_f")
plt.plot(keys, n2, label="SFS_b")
plt.plot(keys, n3, label="SFFS")
plt.plot(keys, n4, label="SFBS")
plt.title("Mean CV Scores vs N# of features")
plt.xlabel("N# features")
plt.ylabel("MSE")
plt.legend()
plt.savefig("FeatureSelection/CVscoresVSfeatures_comparison.png", bbox_inches="tight")
