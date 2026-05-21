# -*- coding: utf-8 -*-
"""
Bivariate analysis for the training set.
Creates the transformed non-encoded dataset used by later steps.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from seaborn import pairplot
from statsmodels.compat import lzip
from statsmodels.graphics.correlation import plot_corr
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera as jb

sns.set()

data = pd.read_csv("DataSets/processed.csv")
data["duration"] = pd.to_timedelta(data["duration"]).dt.total_seconds() / 60.0

cat_vars = [col for col in data.columns if data[col].dtype.kind not in "biufcm"]
cat_vars.remove("reporter")
num_vars = [col for col in data.columns if data[col].dtype.kind in "biufc"]
num_vars.remove("duration")
target = data.duration

for path in [
    "BivariateAnalysis/Numerical",
    "BivariateAnalysis/FullClasses",
    "BivariateAnalysis/ReducedClasses",
]:
    if not os.path.exists(path):
        os.makedirs(path)

# Target distribution
print("\nPlot Histogram for target variable:")
plt.figure(figsize=(12, 8))
data["duration"].hist(bins=range(0, 200, 1))
plt.ylabel("N# of observations")
plt.xlabel("First 200 time classes [1 minute]")
plt.savefig("ExploratoryAnalysis/histogram2.png", bbox_inches="tight")
plt.show()

print("\nBoxPlot for target variable:")
plt.figure(figsize=(12, 8))
data["duration"].plot.box()
plt.savefig("ExploratoryAnalysis/plotbox.png", bbox_inches="tight")
plt.show()

# categorical variables
for c in cat_vars + ["reporter"]:
    confusion_matrix = (
        data.groupby(["duration", c])
        .size()
        .sort_values(ascending=False)
        .unstack(fill_value=0)
    )
    confusion_matrix = confusion_matrix.reindex(
        confusion_matrix.sum().sort_values(ascending=False).index, axis=1
    )
    confusion_matrix = confusion_matrix / confusion_matrix.sum()
    print("####################################################################################################")
    print("\n\033[1m" + "Analisi bivariata duration - " + c + "\033[0;0m \n")
    confusion_matrix.plot.line(title="Analisi bivariata duration - " + c, figsize=(8, 8))
    plt.savefig(f"BivariateAnalysis/ReducedClasses/durationVS{c}.png")
    data.boxplot(column="duration", by=c, figsize=(8, 8), rot=45)
    plt.savefig(f"BivariateAnalysis/ReducedClasses/durationVS{c}_box.png")
    plt.show()

# numerical variables
for c in num_vars:
    plt.figure(figsize=(12, 8))
    plt.title(f"{c} vs. duration", fontsize=16)
    plt.scatter(x=data[c], y=target, color="blue", edgecolor="k")
    plt.grid(True)
    plt.xlabel(c, fontsize=14)
    plt.ylabel("Alert Duration [minutes]", fontsize=14)
    plt.savefig(f"BivariateAnalysis/Numerical/durationVS{c}")
    plt.show()

num_for_pairplot = num_vars + ["duration"]
pairplot(data[num_for_pairplot])
plt.title("Pairplot for numerical features")
plt.savefig("BivariateAnalysis/Numerical/pairplot_num_vars.png")

num_vars_no_target = num_vars.copy()
corr = data[num_vars_no_target].corr()
fig = plot_corr(corr, xnames=corr.columns)
plt.savefig("BivariateAnalysis/Numerical/heatmap.png")

name = ["Jarque-Bera", "Chi^2 two-tail probability", "Skewness", "Kurtosis"]
test_results = jb(data.duration)
lzip(name, test_results)

data.vote_count = np.log(data.vote_count + 1)
data.comment_count = np.log(data.comment_count + 1)
data.description_length = np.log(data.description_length + 1)
data.watch_count = np.log(data.watch_count + 1)
data.duration = np.log(data.duration)

test_results = jb(data.duration)
lzip(name, test_results)

pairplot(data[num_for_pairplot])
plt.title("Pairplot for transformed numerical features")
plt.savefig("BivariateAnalysis/Numerical/pairplot_num_vars_log.png")

data["duration"].hist()
plt.ylabel("N# of observations")
plt.xlabel("Log(Time)")
plt.title("Log-transformed target variable")
plt.savefig("ExploratoryAnalysis/duration_log.png", bbox_inches="tight")
plt.show()

data.to_csv("DataSets/transformed_nonencoded.csv", index=False)

