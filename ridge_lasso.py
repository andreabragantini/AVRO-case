# -*- coding: utf-8 -*-
"""
Ridge and lasso regression on the selected features.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from avro_common import SELECTED_FEATURES

data = pd.read_csv("DataSets/encoded.csv")
x = data[SELECTED_FEATURES]
y = data["duration"]

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
print("R-squared score (training): {:.3f}".format(linridge.score(X_train_scaled, y_train)))
print("R-squared score (test): {:.3f}".format(linridge.score(X_test_scaled, y_test)))
print("Number of non-zero features: {}".format(np.sum(linridge.coef_ != 0)))

trainingRsquared = []
testRsquared = []
print("Ridge regression: effect of alpha regularization parameter\n")
for this_alpha in [0.001, 0.01, 0.1, 0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha=this_alpha).fit(X_train_scaled, y_train)
    trainingRsquared.append(linridge.score(X_train_scaled, y_train))
    testRsquared.append(linridge.score(X_test_scaled, y_test))

alpha = [0.001, 0.01, 0.1, 0, 1, 10, 20, 50, 100, 1000]
trainingRsquared = pd.DataFrame(trainingRsquared, index=alpha)
testRsquared = pd.DataFrame(testRsquared, index=alpha)
df3 = pd.concat([trainingRsquared, testRsquared], axis=1)
df3.columns = ["trainingRsquared", "testRsquared"]
fig5 = df3.plot(figsize=(12, 8), marker=".")
fig5 = plt.title("Ridge training and test squared error vs Alpha")
fig5 = plt.xlabel("alpha")
fig5 = plt.ylabel("SE")
fig5.figure.savefig("FeatureSelection/fig5.png", bbox_inches="tight")

n_alphas = 200
alphas = np.logspace(0, 8, n_alphas)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.figure.set_size_inches(12, 8)
ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.savefig("FeatureSelection/fig6.png", bbox_inches="tight")

linlasso = Lasso(alpha=1000, max_iter=10000).fit(X_train_scaled, y_train)
print("Non-zero features: {}".format(np.sum(linlasso.coef_ != 0)))
print("R-squared score (training): {:.3f}".format(linlasso.score(X_train_scaled, y_train)))
print("R-squared score (test): {:.3f}\n".format(linlasso.score(X_test_scaled, y_test)))

print("Features with non-zero weight (sorted by absolute magnitude):")
for e in sorted(list(zip(list(x), linlasso.coef_)), key=lambda e: -abs(e[1])):
    if e[1] != 0:
        print("\t{}, {:.3f}".format(e[0], e[1]))

trainingRsquared = []
testRsquared = []
for alpha in [0.01, 0.07, 0.05, 0.1, 1, 2, 3, 5, 10, 100, 1000, 10000]:
    linlasso = Lasso(alpha, max_iter=1000000).fit(X_train_scaled, y_train)
    trainingRsquared.append(linlasso.score(X_train_scaled, y_train))
    testRsquared.append(linlasso.score(X_test_scaled, y_test))

alpha = [0.01, 0.07, 0.05, 0.1, 1, 2, 3, 5, 10, 100, 1000, 10000]
trainingRsquared = pd.DataFrame(trainingRsquared, index=alpha)
testRsquared = pd.DataFrame(testRsquared, index=alpha)
df3 = pd.concat([trainingRsquared, testRsquared], axis=1)
df3.columns = ["trainingRsquared", "testRsquared"]
fig7 = df3.plot(figsize=(12, 8), marker=".")
fig7 = plt.title("LASSO training and test squared error vs Alpha")
fig7 = plt.xlabel("alpha")
fig7 = plt.ylabel("SE")
fig7.figure.savefig("FeatureSelection/fig7.png", bbox_inches="tight")

print("Computing regularization path using the LARS ...")
alphas, _, coefs = linear_model.lars_path(X_train_scaled, y_train, method="lasso", verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

fig8 = plt.figure(figsize=(12, 8))
fig8 = plt.plot(xx, coefs.T, marker=".")
ymin, ymax = plt.ylim()
fig8 = plt.vlines(xx, ymin, ymax, linestyle="dashed")
fig8 = plt.xlabel("|coef| / max|coef|")
fig8 = plt.ylabel("Coefficients")
fig8 = plt.title("LASSO Path - Coefficient Shrinkage vs L1")
fig8 = plt.axis("tight")
plt.savefig("FeatureSelection/fig8.png", bbox_inches="tight")

