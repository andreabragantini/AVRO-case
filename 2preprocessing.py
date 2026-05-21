# -*- coding: utf-8 -*-
"""
Preprocessing step:
- split raw data into training and validation
- build the target duration
- remove leakage columns
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from avro_common import PROCESSED_DATA_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH
from avro_common import add_duration, load_raw_dataset, split_train_validation


def main():
    dataset = load_raw_dataset()
    trainset, validationset = split_train_validation(dataset)

    if not os.path.exists("DataSets"):
        os.makedirs("DataSets")
    if not os.path.exists("ExploratoryAnalysis"):
        os.makedirs("ExploratoryAnalysis")

    trainset = add_duration(trainset)
    trainset["description_length"] = trainset["description_length"].fillna(0)

    trainset.to_csv(TRAIN_DATA_PATH, index=False)
    validationset.to_csv(VALIDATION_DATA_PATH, index=False)

    processed = trainset.drop(
        [
            "project",
            "updated",
            "created",
            "resolutiondate",
            "key",
            "days_in_current_status",
            "assignee",
            "status",
            "resolution",
        ],
        axis=1,
    )
    processed.to_csv(PROCESSED_DATA_PATH, index=False)

    duration_minutes = pd.to_timedelta(trainset["duration"]).dt.total_seconds() / 60.0

    print("\nPlot Histogram for target variable:")
    plt.figure(figsize=(12, 8))
    duration_minutes.hist(bins=range(0, 200, 1))
    plt.ylabel("N# of observations")
    plt.xlabel("First 200 time classes [1 minute]")
    plt.savefig("ExploratoryAnalysis/histogram.png", bbox_inches="tight")
    plt.show()

    print("\nBoxPlot for target variable:")
    plt.figure(figsize=(12, 8))
    duration_minutes.plot.box()
    plt.savefig("ExploratoryAnalysis/plotbox.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
