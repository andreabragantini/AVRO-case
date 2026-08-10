import numpy as np
import pandas as pd

from sklearn import metrics as sklearn_metrics

RAW_DATA_PATH = "data_sets/raw/avro-issues.csv"
MERGED_DATA_PATH = "data_sets/raw/avro-issues-merged.csv"
TRAIN_DATA_PATH = "data_sets/dataset.csv"
PROCESSED_DATA_PATH = "data_sets/processed.csv"
TRANSFORMED_DATA_PATH = "data_sets/transformed_nonencoded.csv"
FORECASTING_PILOT_DATA_PATH = "data_sets/forecasting_pilot.csv"
ENCODED_DATA_PATH = "data_sets/encoded.csv"

NUMERIC_LOG_COLUMNS = [
    "description_length",
]

CAT_COLUMNS = ["priority", "issue_type", "reporter"]
SHORT_ISSUE_TYPES = ["Bug", "Improvement", "Task", "Test"]


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def compute_model_metrics(actual_days, predicted_days, r2_log=None):
    """Compute the cross-model comparable metrics on the original day scale.

    The result is a dict so that regression, tree and survival scripts can all
    write the exact same tabular output (see write_metrics_table).

    - c_index: concordance index between predicted and actual resolution time
      (all rows treated as observed events; higher = better ordering).
    - mae_days / median_ae_days: error between predicted and actual days.
    - r2_log: R-squared on the log(minutes) scale (regression/trees only;
      pass None for survival models).
    """
    actual = np.asarray(actual_days, dtype=float)
    predicted = np.asarray(predicted_days, dtype=float)
    c_index = _concordance_index(actual, predicted)
    return {
        "c_index": c_index,
        "mae_days": sklearn_metrics.mean_absolute_error(actual, predicted),
        "median_ae_days": float(np.median(np.abs(actual - predicted))),
        "r2_log": r2_log,
    }


def _concordance_index(actual, predicted):
    try:
        from sksurv.metrics import concordance_index_censored

        event = np.ones(len(actual), dtype=bool)
        # sksurv expects a risk score (higher = shorter time to event); predicted
        # days are a time (higher = longer), so negate them.
        c_index, *_ = concordance_index_censored(event, actual, -np.asarray(predicted, dtype=float))
        return c_index
    except ImportError:
        actual, predicted = np.array(actual), np.array(predicted)
        if len(actual) < 2:
            return np.nan
        pairs = 0
        concordant = 0
        for i in range(len(actual)):
            for j in range(i + 1, len(actual)):
                if actual[i] == actual[j]:
                    continue
                pairs += 1
                pred_i, pred_j = predicted[i], predicted[j]
                if pred_i == pred_j:
                    concordant += 0.5
                elif (pred_i > pred_j) == (actual[i] > actual[j]):
                    concordant += 1
        return concordant / pairs if pairs else np.nan


def write_metrics_table(path, rows, title=""):
    """Write a small, easy-to-open text table of model metrics.

    rows is a list of (model_name, metrics_dict) where metrics_dict comes from
    compute_model_metrics. Every model directory uses the same helper so the
    outputs share an identical structure and can be compared side by side.
    """
    from tabulate import tabulate

    table = []
    for name, m in rows:
        r2 = "n/a" if m["r2_log"] is None else "{:.3f}".format(m["r2_log"])
        table.append(
            [
                name,
                "{:.3f}".format(m["c_index"]),
                "{:.1f}".format(m["mae_days"]),
                "{:.1f}".format(m["median_ae_days"]),
                r2,
            ]
        )
    text = tabulate(
        table,
        headers=["model", "c_index", "mae_days", "median_ae_days", "r2_log"],
        tablefmt="grid",
    )
    if title:
        text = title + "\n\n" + text
    with open(path, "w") as f:
        f.write(text + "\n")
    print("\nWrote metrics table -> {}".format(path))

SELECTED_FEATURES = [
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
]


def load_raw_dataset():
    return pd.read_csv(MERGED_DATA_PATH)


def split_train_forecasting_pilot(dataset):
    train_mask = dataset["resolutiondate"].notna() & dataset["created"].notna()
    trainset = dataset.loc[train_mask].copy().reset_index(drop=True)
    forecasting_pilot = dataset.loc[~train_mask].copy().reset_index(drop=True)
    return trainset, forecasting_pilot


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


def prepare_forecasting_pilot_features(forecasting_pilot, training_columns):
    frame = forecasting_pilot.copy()
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
