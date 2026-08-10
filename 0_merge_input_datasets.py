# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 2026

@author: andre
INPUT DATASETS - JSON
Reads the nested avro-issues.json export, extracts flat predictive features and
merges them with the raw avro-issues.csv export.

This script MUST run first (it is the first entry in run_analysis.py). It does
NOT modify the raw avro-issues.csv; instead it writes a NEW dataset,
data_sets/raw/avro-issues-merged.csv, that every downstream script
(exploratory analysis, preprocessing, bivariate analysis, encoding, models)
reads as its input.
"""

import pandas as pd

import utils.json_helpers
from utils.avro_common import print_section

print_section("0. INPUT DATASETS - JSON")

RAW_CSV_PATH = "data_sets/raw/avro-issues.csv"
MERGED_CSV_PATH = "data_sets/raw/avro-issues-merged.csv"

# Only features that are known (or mostly known) at issue-creation time are
# merged. The time-variant ones (fix versions, attachments, comments, first
# response delay) are intentionally left out of the training features; they are
# explored in 2_explore_json.py instead.
FEATURES_TO_MERGE = [
    "issue_number",
    "num_components",
    "component_java",
    "num_affected_versions",
    "num_labels",
]

# 1) Build the feature frame from the JSON (indexed by issue key)
features = utils.json_helpers.build_feature_frame()
features = features[FEATURES_TO_MERGE]
print(f"\nExtracted {len(features)} rows and {len(features.columns)} features "
      f"from {utils.json_helpers.JSON_DATA_PATH}")

# 2) Load the raw CSV (left pristine) and merge on the issue key
dataset = pd.read_csv(RAW_CSV_PATH)
dataset = dataset.merge(
    features, left_on="key", right_index=True, how="left"
)

# 3) Calendar features derived from 'created' (safe, known at creation)
created = pd.to_datetime(dataset["created"], errors="coerce")
dataset["created_weekday"] = created.dt.dayofweek
dataset["created_month"] = created.dt.month

# 4) Save to the NEW merged dataset (the raw file is never rewritten)
dataset.to_csv(MERGED_CSV_PATH, index=False)
print(f"\nWrote {MERGED_CSV_PATH} with JSON-derived features:")
for col in FEATURES_TO_MERGE + ["created_weekday", "created_month"]:
    print(f"  {col}: non-null={dataset[col].notna().sum()}, "
          f"mean={dataset[col].mean():.3f}")
