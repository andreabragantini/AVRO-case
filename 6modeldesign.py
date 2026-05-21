# -*- coding: utf-8 -*-
"""
Multiple linear regression model.
Uses the feature subset selected during feature selection.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import qqplot

from avro_common import SELECTED_FEATURES

data = pd.read_csv("DataSets/encoded.csv")
predictors = data.iloc[:, :-1]

if not os.path.exists("ModelDesign"):
    os.makedirs("ModelDesign")

X = predictors[SELECTED_FEATURES]
y = data["duration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

print("R-squared score (training): {:.3f}".format(linreg.score(X_train, y_train)))
print("R-squared score (test): {:.3f}".format(linreg.score(X_test, y_test)))

y_pred = linreg.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

formula_str = data.columns[-1] + " ~ " + "+".join(SELECTED_FEATURES)
model = sm.ols(formula=formula_str, data=pd.concat([X_train, y_train], axis=1))
fitted = model.fit()
print(fitted.summary())

coef = pd.DataFrame(fitted.params, columns=["value"])
df_result = pd.DataFrame()
df_result["pvalues"] = fitted.pvalues[1:]
df_result["Features"] = SELECTED_FEATURES
df_result.set_index("Features", inplace=True)
df_result["Statistically significant?"] = df_result["pvalues"].apply(
    lambda value: "Yes" if value < 0.05 else "No"
)
print(df_result)

for c in SELECTED_FEATURES:
    plt.figure(figsize=(12, 8))
    plt.title(f"{c} vs. Model residuals", fontsize=16)
    plt.scatter(x=X_train[c], y=fitted.resid, color="blue", edgecolor="k")
    plt.grid(True)
    plt.hlines(y=0, xmin=min(data[c]) * 0.9, xmax=max(data[c]) * 1.1, color="red", linestyle="--", lw=3)
    plt.xlabel(c, fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(x=fitted.fittedvalues, y=fitted.resid, edgecolor="k")
plt.hlines(
    y=0,
    xmin=min(fitted.fittedvalues) * 0.9,
    xmax=max(fitted.fittedvalues) * 1.1,
    color="red",
    linestyle="--",
    lw=3,
)
plt.xlabel("Fitted values", fontsize=15)
plt.ylabel("Residuals", fontsize=15)
plt.title("Fitted vs. residuals plot", fontsize=18)
plt.grid(True)
plt.savefig("ModelDesign/resVSfit.png")
plt.show()

plt.figure(figsize=(12, 8))
plt.hist(fitted.resid_pearson, bins=20, edgecolor="k")
plt.ylabel("Count", fontsize=15)
plt.xlabel("Normalized residuals", fontsize=15)
plt.title("Histogram of normalized residuals", fontsize=18)
plt.savefig("ModelDesign/resHist.png")
plt.show()

plt.figure(figsize=(12, 8))
qqplot(fitted.resid_pearson, line="45", fit="True")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Theoretical quantiles", fontsize=15)
plt.ylabel("Sample quantiles", fontsize=15)
plt.title("Q-Q plot of normalized residuals", fontsize=18)
plt.grid(True)
plt.savefig("ModelDesign/resQQplot.png")
plt.show()

_, p = shapiro(fitted.resid)
if p > 0.01:
    print("The residuals seem compatible with a Gaussian process")
else:
    print("The normality assumption may not hold")

y_pred = fitted.predict(X_test)
print("R-squared score (test): {:.3f}".format(r2_score(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

df_ols = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
df_ols.reset_index(drop=True).plot(figsize=(12, 8), marker=".")
plt.xlabel("Test Set observations", fontsize=15)
plt.ylabel("LOG(Duration)", fontsize=15)
plt.title("Predicted vs. TestSet", fontsize=18)
plt.savefig("ModelDesign/predictedVStest.png")
plt.show()

df_ols = np.exp(df_ols)
df_ols.reset_index(drop=True).plot(figsize=(12, 8), marker=".")
plt.xlabel("Test Set observations", fontsize=15)
plt.ylabel("Duration", fontsize=15)
plt.title("Predicted vs. TestSet", fontsize=18)
plt.savefig("ModelDesign/predictedVStest_transf.png")
plt.show()

