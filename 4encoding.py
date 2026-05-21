# -*- coding: utf-8 -*-
"""
One-hot encoding for the transformed dataset.
"""

import pandas as pd

data = pd.read_csv("DataSets/transformed_nonencoded.csv")

cat_vars = ["priority", "issue_type", "reporter"]
data = pd.get_dummies(data, columns=cat_vars, drop_first=True)
data = data[[c for c in data if c != "duration"] + ["duration"]]

data.to_csv("DataSets/encoded.csv", index=False)

