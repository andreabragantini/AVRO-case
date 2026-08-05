"""Run the analysis scripts in a single shared Python session.

This keeps the workflow simple: later scripts can reuse models and variables
defined by earlier ones, which matches how the original notebooks were used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib


# Use a non-interactive backend so the pipeline can run end-to-end without
# waiting for figures to be dismissed.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

OUTPUT_DIRS = [
    "about",
    "bivariate_analysis",
    "bivariate_analysis/numerical",
    "bivariate_analysis/full_classes",
    "bivariate_analysis/reduced_classes",
    "data_sets",
    "feature_selection",
    "multi_lin_reg",
    "question2",
    "regression_tree",
]

# Core analysis chain.  The JSON inspection script (2_explore_json.py) is
# intentionally left out because it is a standalone helper and does not feed
# the main workflow.
SCRIPTS = [
    "0_merge_input_datasets.py",
    "1_exploratory_analysis.py",
    "2_preprocessing.py",
    "3_bivariate_analysis.py",
    "4_encoding.py",
    "5_feature_selection.py",
    "ridge_lasso.py",
    "6_multi_lin_reg.py",
    "7_regression_trees.py",
    "9_predicting.py",
]


def run_script(path: Path, namespace: dict[str, object]) -> None:
    print(f"\n=== Running {path.name} ===")
    namespace["__file__"] = str(path)
    namespace["__name__"] = "__main__"
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), namespace)
    plt.close("all")


def main() -> None:
    namespace: dict[str, object] = {"__name__": "__main__"}
    for output_dir in OUTPUT_DIRS:
        (ROOT / output_dir).mkdir(parents=True, exist_ok=True)
    for script_name in SCRIPTS:
        run_script(ROOT / script_name, namespace)
    print("\nAnalysis pipeline completed.")


if __name__ == "__main__":
    main()
