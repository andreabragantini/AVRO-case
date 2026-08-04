import numpy as np
import pandas as pd

RAW_DATA_PATH = "data_sets/raw/avro-issues.csv"
TRAIN_DATA_PATH = "data_sets/dataset.csv"
PROCESSED_DATA_PATH = "data_sets/processed.csv"
TRANSFORMED_DATA_PATH = "data_sets/transformed_nonencoded.csv"
VALIDATION_DATA_PATH = "data_sets/validationset.csv"
ENCODED_DATA_PATH = "data_sets/encoded.csv"

NUMERIC_LOG_COLUMNS = [
    "vote_count",
    "comment_count",
    "description_length",
    "watch_count",
]

CAT_COLUMNS = ["priority", "issue_type", "reporter"]
SHORT_ISSUE_TYPES = ["Bug", "Improvement", "Task", "Test"]


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

SELECTED_FEATURES = [
    "comment_count",
    "issue_type_Short",
    "priority_Minor",
    "reporter_cutting",
    "reporter_dcreager",
    "reporter_hammer",
    "reporter_massie",
    "reporter_sbanacho",
    "reporter_scott_carey",
    "reporter_sharadag",
    "reporter_tomwhite",
    "reporter_vnadkarni",
    "vote_count",
    "watch_count",
]


def load_raw_dataset():
    return pd.read_csv(RAW_DATA_PATH)


def split_train_validation(dataset):
    train_mask = dataset["resolutiondate"].notna() & dataset["created"].notna()
    trainset = dataset.loc[train_mask].copy().reset_index(drop=True)
    validationset = dataset.loc[~train_mask].copy().reset_index(drop=True)
    return trainset, validationset


def add_duration(dataset):
    frame = dataset.copy()
    created = pd.to_datetime(frame["created"], errors="coerce")
    resolutiondate = pd.to_datetime(frame["resolutiondate"], errors="coerce")
    frame["duration"] = resolutiondate - created
    return frame


def reduce_categories(dataset, reporter_min_count=10):
    frame = dataset.copy()
    reporter_counts = frame["reporter"].value_counts()
    selected_reporters = reporter_counts[reporter_counts > reporter_min_count].index
    frame["reporter"] = frame["reporter"].map(
        lambda value: value if value in selected_reporters else "Other"
    )
    # NOTE: the Short/Long split below was chosen in the bivariate analysis
    # after observing which issue types tend to have short vs long resolution
    # times (3_bivariate_analysis.py). That makes it mildly target-informed;
    # it is kept as a simplification and documented here for transparency.
    frame["issue_type"] = frame["issue_type"].map(
        lambda value: "Short" if value in SHORT_ISSUE_TYPES else "Long"
    )
    return frame


def log_transform_numeric_features(dataset):
    frame = dataset.copy()
    for column in NUMERIC_LOG_COLUMNS:
        frame[column] = np.log(frame[column] + 1)
    return frame


def prepare_validation_features(validation_dataset, training_columns):
    frame = validation_dataset.copy()
    frame = frame.drop(
        [
            "project",
            "updated",
            "created",
            "resolutiondate",
            "key",
            "days_in_current_status",
            "assignee",
            "resolution",
            "status",
        ],
        axis=1,
    )

    frame["description_length"] = frame["description_length"].fillna(0)

    known_reporters = {
        column.replace("reporter_", "")
        for column in training_columns
        if column.startswith("reporter_")
    }
    frame["reporter"] = frame["reporter"].map(
        lambda value: value if value in known_reporters else "Other"
    )
    frame["issue_type"] = frame["issue_type"].map(
        lambda value: "Short" if value in SHORT_ISSUE_TYPES else "Long"
    )
    frame = log_transform_numeric_features(frame)
    frame = pd.get_dummies(frame, columns=CAT_COLUMNS, drop_first=True)

    for column in training_columns:
        if column not in frame.columns:
            frame[column] = 0

    extra_columns = [column for column in frame.columns if column not in training_columns]
    if extra_columns:
        frame = frame.drop(extra_columns, axis=1)

    return frame[training_columns]
