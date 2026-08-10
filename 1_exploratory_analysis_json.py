# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:58:01 2020

@author: andre
EXPLORE JSON FILE
Standalone inspection of the raw Apache Avro JIRA export (avro-issues.json).

This script is NOT part of the main analysis pipeline. It exists to understand
the nested JSON structure and to evaluate which fields can be turned into
predictive features. It prints an availability report and saves a few
distribution plots to exploratory_analysis/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.json_helpers

if not os.path.exists("exploratory_analysis"):
    os.makedirs("exploratory_analysis")

issues = utils.json_helpers.load_issues()
print(f"Number of issues: {len(issues)}")

# 1) Structure of a single issue
first = issues[0]
print("\nTop-level keys of one issue:", sorted(first.keys()))
print("Fields keys:", sorted(first["fields"].keys()))
print(
    "Example issue:", first["key"],
    "| status:", first["fields"]["status"]["name"],
    "| issue_type:", first["fields"]["issuetype"]["name"],
)

# 2) Field availability across the whole dataset
features = utils.json_helpers.build_feature_frame()
features = features.replace([np.inf, -np.inf], np.nan)
availability = pd.DataFrame(
    {
        "non_null": features.notna().sum(),
        "fill_rate": features.notna().mean(),
    }
).sort_values("fill_rate", ascending=False)
print("\nField availability across all issues:")
print(availability.round(3).to_string())

# 3) Components distribution
comp_counts = {}
for issue in issues:
    for name in utils.json_helpers.component_names(issue):
        comp_counts[name] = comp_counts.get(name, 0) + 1
top_components = sorted(comp_counts.items(), key=lambda kv: kv[1], reverse=True)
print("\nTop components:")
for name, count in top_components[:12]:
    print(f"  {name}: {count}")

# 4) Distributions of the candidate numeric features
for col in [
    "num_components",
    "num_affected_versions",
    "num_fix_versions",
    "num_labels",
    "attachment_count",
    "distinct_comment_authors",
]:
    series = features[col].dropna()
    print(
        f"{col}: min={series.min():g}, median={series.median():g}, "
        f"max={series.max():g}"
    )

first_response = features["first_response_delay_hours"].dropna()
print(
    "\nfirst_response_delay_hours: n={}, median={:.2f} h, mean={:.2f} h".format(
        len(first_response), first_response.median(), first_response.mean()
    )
)

# 5) Plots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
count_cols = [
    "num_components",
    "num_affected_versions",
    "num_fix_versions",
    "num_labels",
    "attachment_count",
    "distinct_comment_authors",
]
for ax, col in zip(axes.ravel(), count_cols):
    features[col].value_counts().sort_index().plot(kind="bar", ax=ax, title=col)
    ax.set_xlabel("value")
plt.tight_layout()
plt.savefig("exploratory_analysis/json_feature_counts.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
components = pd.Series(dict(top_components[:12]))
components.plot(kind="barh", ax=axes[0], title="Top components")
axes[0].invert_yaxis()
axes[0].set_xlabel("number of issues")
np.log1p(first_response).hist(bins=40, ax=axes[1])
axes[1].set_title("log(1 + first_response_delay_hours)")
axes[1].set_xlabel("log(hours)")
plt.tight_layout()
plt.savefig("exploratory_analysis/json_components_and_first_response.png")
plt.close()
print("\nSaved JSON exploration plots to exploratory_analysis/.")
