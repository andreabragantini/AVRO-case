# AVRO-case

This repository contains a **data science exercise** built around the public issue tracker
of **Apache Avro**, an open-source data serialization framework (a bit like JSON or
Protobuf) used mainly in big-data systems. The repository is deliberately kept
**educational**: every step is a small, well-commented Python script, from data
munging to survival analysis, so that a beginner can follow the whole journey
of a machine-learning project from raw data to final conclusions.

---

## 1. At a glance

- **The question.** Given the information available when a new issue is opened,
  can we predict **how long it will take to resolve it**?
- **The data.** A snapshot of **1,458 issues** from the Apache Avro JIRA tracker
  (who reported it, priority, type, votes, comments, timestamps, ...).
- **The split.** **1,134 resolved issues** are used for training; the other
  **324 issues were still open** at snapshot time and are kept as a
  *forecasting pilot* (a sanity check on realistic cases) — they have no
  resolution date, hence no ground truth.
- **The models.** Five model families are compared: **multiple linear
  regression**, **decision tree**, **random forest**, **Cox proportional
  hazards**, and **random survival forest**.
- **The focus.** Simple, explainable models that can be inspected and
  understood — not squeezing out the last bit of accuracy.
- **The stack.** Python with pandas, scikit-learn, mlxtend, statsmodels,
  lifelines, scikit-survival, matplotlib and seaborn.

The headline result in one line: the models answer *"how long does a typical
issue take?"* reliably (median absolute error ~5 days), but they systematically
miss the rare, very long-running issues — a documented limitation discussed in
[Section 11](#11-discussion-why-the-predictions-look-too-small).

---

## 2. The data

### 2.1 Where it comes from

The raw data lives in `data_sets/raw/` as two exports of the same tracker:

- `avro-issues.csv` — a flat table of all 1,458 issues.
- `avro-issues.json` — the nested JIRA REST export (one JSON object per line).

The first pipeline script (`0_merge_input_datasets.py`) extracts flat, useful
features from the nested JSON (`num_components`, `component_java`,
`num_affected_versions`, `num_labels`, `issue_number`) and **merges** them into
the CSV, writing `data_sets/raw/avro-issues-merged.csv`. The raw CSV is never
modified.

The raw dataset contains 1,458 issues. Not all of them can be used to train a
prediction model, because only issues with a valid creation date and resolution
date can tell us the real closing time.

### 2.2 The target variable

The thing we want to predict is the **resolution time** — the time between when
an issue was created (`created`) and when it was resolved
(`resolutiondate`):

```
duration = resolutiondate - created
```

Because durations are extremely skewed (a few issues take months or years),
the models are trained on the **logarithm** of the duration, in **minutes**
(`log(minutes)`). Predictions are transformed back with the exponential for
reporting on the original day scale. *Why the log is important is explained in
depth in [Section 11](#11-discussion-why-the-predictions-look-too-small).*

### 2.3 The train / forecasting-pilot split

| Set | Rows | What it is |
| --- | ---: | --- |
| Training | 1,134 | Issues with a valid `created` **and** `resolutiondate` — the only ones that give us a real closing time. |
| Forecasting pilot | 324 | Issues still open at snapshot time; no ground truth. Used in `9_predicting.py` as a qualitative check on realistic, still-open cases. |

Quantitative evaluation is always done on a **random 80/20 train/test split of
the 1,134 resolved issues** (seed 0), not on the pilot.

### 2.4 Data flow (who writes what)

| Script | Writes | Content |
| --- | --- | --- |
| `0_merge_input_datasets.py` | `data_sets/raw/avro-issues-merged.csv` | CSV + JSON features merged |
| `2_preprocessing.py` | `data_sets/dataset.csv` | training set (resolved issues only) |
| `2_preprocessing.py` | `data_sets/forecasting_pilot.csv` | the 324 still-open issues |
| `2_preprocessing.py` | `data_sets/processed.csv` | training set minus leaky/unusable columns |
| `3_bivariate_analysis.py` | `data_sets/transformed_nonencoded.csv` | log-transformed features + log target |
| `4_encoding.py` | `data_sets/encoded.csv` | one-hot encoded, 32 features + target (1,134 × 33) |

---

## 3. Repository layout

```
AVRO-case/
├── run_analysis.py            # one-shot runner: executes all scripts in order
├── 0_merge_input_datasets.py  # JSON -> flat features -> merged CSV
├── 1_exploratory_analysis.py  # distributions, missing values, first plots
├── 1_exploratory_analysis_json.py  # (standalone) inspect the raw JSON export
├── 2_preprocessing.py         # target variable, train/pilot split, drop leaks
├── 3_bivariate_analysis.py    # target vs each predictor, log-transforms
├── 4_encoding.py              # one-hot encoding -> encoded.csv
├── 5_feature_selection.py     # 4 sequential feature-selection techniques
├── 5_ridge_lasso.py           # Ridge/Lasso cross-check on the selection
├── 6_multi_lin_reg.py         # multiple linear regression model
├── 7_regression_trees.py      # decision tree + random forest
├── 8_survival_analysis.py     # Cox PH + random survival forest
├── 9_predicting.py            # forecasting pilot on the 324 open issues
├── 10_model_comparison.py     # final side-by-side model comparison
├── requirements.txt           # package dependencies
├── data_sets/                 # raw + intermediate + final datasets (CSV)
├── exploratory_analysis/      # EDA plots and dataset info
├── bivariate_analysis/        # target-vs-predictor plots
├── feature_selection/         # selection results (plots, joblib, summary)
├── multi_lin_reg/             # linear model artifacts
├── regression_tree/           # tree model artifacts
├── survival_analysis/         # survival model artifacts
├── question2/                 # forecasting-pilot predictions & report
├── results/                   # final model comparison
├── utils/                     # shared helpers (avro_common.py, json_helpers.py)
└── about/                     # the original exercise description (PDF/PPTX)
```

A **compact appendix** at the end lists the most important files in each output
folder and what they are.

---

## 4. Main findings

These are the broad, take-away conclusions. The detailed evidence lives in the
deeper sections (pipeline, models, discussion).

1. **Resolution times are extremely skewed.** Most issues close within ~6 days,
   but the average is ~47 days and a few issues run for months or years
   (longest ~952 days). A handful of outliers dominate the mean.
2. **Several columns cannot be used to forecast a new issue.** `status`,
   `resolution`, `assignee` and the date fields leak the answer or are unknown
   at creation time, so they are removed. The activity counts
   (`vote_count`, `comment_count`, `watch_count`) grow *with* the age of an
   issue, so they are circular and are dropped too.
3. **Features extracted from the raw JSON add real signal.** `num_components`,
   `num_affected_versions` and `num_labels` are chosen by every feature-selection
   technique, together with `issue_type` and a few frequent reporters.
4. **The linear model is interpretable but modest** (training R² ≈ 0.18, test
   R² ≈ 0.13). It explains part of the variance but not all of it.
5. **Tree models are more robust on extreme cases, but overfit more.** The
   single decision tree overfits easily; the random forest is safer for
   prediction, yet it also fails to catch the very long issues.
6. **Survival models are the only ones that use all 1,458 issues.** By treating
   the 324 still-open issues as censored observations, Cox PH and the random
   survival forest can answer the practical question *"will this still be open
   in three months?"* instead of only the typical duration.

Minor, more specific findings are pointed out where they belong — e.g. the
feature-selection winner and the final 7-feature set are in
[Section 8](#8-feature-selection-in-depth), and the long-tail behavior in
[Section 11](#11-discussion-why-the-predictions-look-too-small).

---

## 5. Getting started

### 5.1 Install

```bash
pip install -r requirements.txt
```

The list includes the JIRA-JSON parser (`ijson`), the feature-selection package
(`mlxtend`), and the two survival libraries (`lifelines`,
`scikit-survival`).

### 5.2 Run everything

From the repository root:

```bash
python run_analysis.py
```

This runs all scripts in a **single shared Python session** so that later
stages reuse models and variables created earlier. It also uses a
non-interactive Matplotlib backend, so every plot is saved to disk instead of
opening a window. Expect a run time of ~7–9 minutes (the feature-selection
step is the slowest).

### 5.3 Run a single script

Every script is **self-contained**: each fitting script persists its outputs to
disk (feature indices in `feature_selection/`, fitted models in
`multi_lin_reg/`, `regression_tree/` and `survival_analysis/`), and the
predicting/comparison scripts load them from those folders instead of relying
on shared-session variables. So you can, for example, run
`python 6_multi_lin_reg.py` alone after the feature-selection scripts have run
once.

> Tip for learning: run the scripts **one at a time** in the order of the
> walkthrough below, and open the outputs after each step — that is the best
> way to see what each stage contributes.

---

## 7. The pipeline, script by script

This section describes each script: its role, its inputs/outputs, and what to
look at. Reading order follows the pipeline.

### 7.0 `0_merge_input_datasets.py` — inputs & JSON features

The JIRA CSV is flat, but the richer data (components, affected versions,
labels) lives in the nested JSON export. This script parses the JSON with the
helpers in `utils/json_helpers.py`, extracts only features that are known at
issue-creation time, and merges them into the CSV.

- **Merged features:** `issue_number` (the numeric part of the key, used as an
  era/trend proxy), `num_components`, `component_java` (flag: has a Java
  component), `num_affected_versions`, `num_labels`.
- **Output:** `data_sets/raw/avro-issues-merged.csv`. The raw
  `avro-issues.csv` is never touched.

### 7.1 `1_exploratory_analysis.py` — exploratory analysis

A first look at every variable: counts and bar charts for categoricals,
histograms for numericals, plus a normality check on the target. This is where
you notice the skewed resolution-time distribution and the structure of
`status`, `priority`, `issue_type`, `reporter`, etc.

- **Outputs:** `exploratory_analysis/datasetInfo.txt` (column report),
  `freq_*.png` bar charts, `histogram.png`, `plotbox.png`, `json_*.png`.

### 7.1b `1_exploratory_analysis_json.py` — standalone JSON inspection

A **standalone helper** (not part of the modeling chain). It prints the
structure of one issue object from the JSON export and evaluates which nested
fields could become predictive features. Useful to understand *why* the merge
in `0_merge_input_datasets.py` picks what it picks.

### 7.2 `2_preprocessing.py` — target, split, and leak removal

This is where the dataset becomes usable for supervised learning:

1. **Split.** `trainset` = issues with a valid `created` and `resolutiondate`
   (1,134 rows); the rest (324) become the forecasting pilot.
2. **Target.** `duration = resolutiondate - created`.
3. **Drop unusable / leaky columns:** `project` (constant), `updated` (happens
   after resolution), `resolutiondate` and `created` (already used for the
   target), `key` (unique per row), `days_in_current_status` (not relevant),
   `assignee` (often missing, and unresolved issues have none), `status`
   (a new issue is never `Closed`/`Resolved` yet), `resolution` (only exists
   after closing). Missing `description_length` is filled with `0`.
4. **Save** `dataset.csv`, `forecasting_pilot.csv` and `processed.csv`.

- **Educational note.** *Why drop `status`?* The training set only contains
  closed issues, so their observed status is the *end* state (`Closed` /
  `Resolved`) — which would trivially leak the outcome. A new issue would never
  have those values.

### 7.3 `3_bivariate_analysis.py` — target vs each predictor

A systematic study of how the target relates to every predictor: scatter plots
and hexbins for numerical features, box/violin plots per class for categoricals
(using the full class lists and the *reduced* ones — see below). It also runs a
normality test on the target and then **log-transforms** the skewed variables
(counts +1, duration without offset) — this is what produces
`data_sets/transformed_nonencoded.csv`.

- **Reduced classes.** Rare reporters are grouped into `Other`, and
  `issue_type` is collapsed to `Short` / `Long` based on typical resolution
  times observed here. See [Section 12](#12-known-limitations) for the caveat
  about this being target-informed.
- **Outputs:** all plots under `bivariate_analysis/numerical/`,
  `full_classes/`, `reduced_classes/`, plus `duration_log.png` and the
  transformed pairplot.

### 7.4 `4_encoding.py` — one-hot encoding

Machine-learning libraries need numbers, so the categorical variables are
one-hot encoded. *One-hot encoding* turns a column like `issue_type` with
values `Bug`/`Improvement`/... into one binary `0/1` column per value; the
`drop_first=True` option drops one redundant column per category to avoid
perfect multicollinearity.

It also **drops the three circular activity counts** (`vote_count`,
`comment_count`, `watch_count`) — they grow with the age of an issue, so they
carry no signal for a brand-new issue and would leak its age.

- **Output:** `data_sets/encoded.csv` — **1,134 rows × 33 columns** (32
  features + the `duration` target in the last column).

### 7.5 `5_feature_selection.py` — sequential feature selection

32 features is a lot for an interpretable linear model. This script runs four
*sequential* search techniques to find a smaller, useful subset. It is the
educational core of the project and gets its own deep-dive in
[Section 8](#8-feature-selection-in-depth).

- **Outputs:** `feature_selection/CVscoresVSfeatures_comparison.png`,
  `best_sequential.joblib` (winning indices),
  `sequential_subsets.joblib` (all four techniques).

### 7.5b `5_ridge_lasso.py` — Ridge/Lasso cross-check

A **diagnostic** script: Ridge and Lasso are regularized linear models used
here not as predictive models but as an **independent second opinion** on the
feature selection. Lasso's L1 penalty can shrink coefficients to exactly zero
(an automatic feature selector), and its kept features are compared against the
sequential winner. Ridge's L2 penalty never zeroes coefficients, so it reports
which features are *consistently strong*.

- **Outputs:** `feature_selection/ridge_*.png`, `lasso_*.png`,
  `feature_selection_summary.txt` (full console transcript),
  `final_features.joblib` (the final feature set used by the linear model).

### 7.6 `6_multi_lin_reg.py` — multiple linear regression

Fits an ordinary-least-squares linear model on the **final feature set** chosen
in the previous steps, plus a statsmodels OLS fit with p-values for
interpretation, and full residual diagnostics (residual-vs-predictor plots,
fitted-vs-residual, histogram, Q-Q plot, Shapiro test).

- **Outputs:** `multi_lin_reg/linreg.joblib`, `model_metrics.txt`,
  `predictedVStest*.png`, and `multi_lin_reg/residuals/`.

### 7.7 `7_regression_trees.py` — decision tree & random forest

Fits a single **decision tree** (depth-limited to keep it readable and avoid
overfitting) and a **random forest** — an ensemble of many trees whose
predictions are averaged. It exports the tree to `tree.dot` (readable in
Graphviz) and plots feature importances.

- **Educational note.** A single tree is easy to inspect but overfits; the
  forest is more robust but harder to interpret. Compare their metrics in
  `regression_tree/model_metrics.txt`.
- **Outputs:** `tree.joblib`, `rfr.joblib`, `tree.dot`,
  `ParametersImportance_rfr.png`, `predictedVStest*.png`.

### 7.8 `8_survival_analysis.py` — Cox PH & random survival forest

A different paradigm. Instead of predicting one number (the duration), *survival
analysis* models the probability of *not yet being resolved* over time, which
lets the model **exploit the 324 still-open issues as censored observations**
instead of discarding them. Two models are fitted:

- **Cox proportional hazards** — a semi-parametric model of how features shift
  the risk of resolution.
- **Random survival forest** — a tree ensemble adapted to censored data.

It also draws **Kaplan-Meier survival curves**, both overall and by
`issue_type` (`Short` vs `Long`).

- **Outputs:** `survival_analysis/km_overall.png`, `km_by_issue_type.png`,
  `cph.joblib`, `rsf.joblib`, `cox_pilot.joblib`, `rsf_pilot.joblib`,
  `survival_eval.joblib`, `model_metrics.txt`, `predictedVStest*.png`.

### 7.9 `9_predicting.py` — forecasting pilot

Applies **all five fitted models to the 324 still-open issues**. Regression and
tree models give the predicted *total* resolution time from creation; the
survival models give the *remaining* days (conditional on how long the issue
has already been open) plus P(resolve in 30/90/180 days).

- **Outputs:** `question2/forecasting_pilot_predictions.csv` (per-issue table),
  `question2/predictionComparison.png` / `_log.png` (all models on the same
  axes), and `question2/forecasting_pilot_results.txt` (interesting cases +
  conclusions).

### 7.10 `10_model_comparison.py` — final comparison

Loads all fitted models from disk and evaluates them **on the same held-out 20%
of the resolved issues**, writing one comparison table. It no longer repeats
the forecasting pilot (that lives in `9_predicting.py`).

- **Output:** `results/model_comparison.txt`.

### 7.11 Supporting files

- `run_analysis.py` — the runner described in
  [Section 5.2](#52-run-everything). It creates every output directory, then
  executes the scripts in order in one Python session.
- `utils/avro_common.py` — shared helpers: section banners, the uniform model
  metrics (`compute_model_metrics` / `write_metrics_table`), the reporter
  `Other`-grouping, the `Short`/`Long` mapping, and the forecasting-pilot
  feature preparation.
- `utils/json_helpers.py` — parses the nested JSON export into flat features
  (used by `0_merge_input_datasets.py` and `1_exploratory_analysis_json.py`).

---

## 8. Feature selection in depth

### 8.1 Why select features at all?

A linear model with all 32 features is still usable, but a smaller subset is
easier to interpret and can generalize better. Feature selection answers:
*"which subset of the available columns actually carries signal?"*

### 8.2 The four sequential techniques (`5_feature_selection.py`)

All four are *wrapper* methods: they score a candidate subset by fitting a
`LinearRegression` and measuring its **5-fold cross-validation error**. They
differ only in how they move through the space of subsets.

- **SFS_f (Sequential Forward Selection)** — start empty, at each step *add*
  the single feature that improves the CV score most. Never removes.
- **SBS (Sequential Backward Selection)** — start with all features, at each
  step *remove* the one whose removal hurts the least.
- **SFFS / SFBS (floating variants)** — like forward/backward, but after each
  step they may *undo* one move (add back / remove again) if that improves the
  score. Floating searches more of the space and usually finds at least as good
  a subset, at extra computational cost.

*Cross-validation* means the data is split into 5 folds and each subset is
scored by fitting on 4 folds and evaluating on the 5th, repeating over all
folds — so the score estimates out-of-sample performance. The scoring is done
with `neg_mean_squared_error` because sklearn's convention is that *higher* is
better; the scripts undo the negation to report a plain MSE where *lower* is
better.

No standardisation is needed here because ordinary least squares is
scale-invariant (multiplying a feature by a constant does not change the
predictions). Standardisation only matters for penalized models — which is why
`5_ridge_lasso.py` scales first.

> **Why not an exhaustive search?** Trying every possible subset means 2³² ≈ 4
> billion models — computationally impossible. Sequential searches walk a tiny,
> tractable fraction of that space.

### 8.3 How the winner is chosen

Each technique picks the feature count `k` with the **lowest mean CV error**,
then the four "best subsets" are compared on the same criterion. Because every
technique uses the *same* deterministic 5-fold split, the four scores are
directly comparable. The winner is the one with the lowest mean CV error; ties
are broken by preferring fewer features.

On the current data the winner is **SBS with 16 features** (mean CV
MSE ≈ 6.031), very closely followed by the two floating variants — all three
select essentially the same 16 features, while plain SFS_f is slightly worse
and prefers a slightly different set. All four selections agree on the core
signal: `issue_type_Short`, `num_components`, `num_affected_versions`,
`num_labels`, `priority_Minor`, and the frequent reporters.

The final intuition is that the sequential search for feature selection does 
not change much its final outcome even though different algorithms are used.

### 8.4 The Ridge/Lasso cross-check (`5_ridge_lasso.py`)

To complete the feature selection step, besides classical sequential approaches,
also ridge and lasso techniques have been applied. This is done mainly to 
verify the results from the sequential features selection and see if the results
were comparable or not.

- **Lasso** (L1 penalty) shrinks some coefficients to *exactly zero*, so its
  kept features (11 on this data) are an independent, automatic selection. It
  agrees with the sequential winner on **7 features** — the overlap is a strong
  signal that those features genuinely matter.
- **Ridge** (L2 penalty) never zeroes coefficients; the strongest ones by
  |standardized coefficient| are `reporter_massie`, `issue_type_Short`,
  `num_labels` and `num_affected_versions`.

### 8.5 The final feature set

Because the two independent routes (greedy sequential vs. penalized Lasso)
agree so strongly, the linear model is built on their **7-feature overlap**
(`feature_selection/final_features.joblib`):

```
num_affected_versions, num_labels, priority_Trivial, issue_type_Short,
reporter_dcreager, reporter_massie, reporter_sbanacho
```

This small set scores best in a 5-fold CV R² cross-check among four candidates:

| Feature set | Features | CV R² (mean) |
| --- | ---: | ---: |
| Lasso & sequential overlap | 7 | **0.146** |
| Lasso | 11 | 0.145 |
| Best sequential (SBS) | 16 | 0.139 |
| All features | 32 | 0.121 |

The overlap wins by a slim margin — and it is much easier to interpret.

---

## 9. The models and how they compare

### 9.1 The metrics (a mini-glossary)

Every model family writes the **same** metrics table so results can be compared
directly. All error metrics are on the original **day scale** (the models train
on `log(minutes)` and the predictions are transformed back).

- **`c_index` (concordance index)** — how well the model *ranks* issues: for
  every pair of issues, does the model predict the longer one to take longer?
  0.5 = random, 1 = perfect. Survival models compute a censored-aware version
  that also includes the 324 still-open issues.
- **`mae_days`** — mean absolute error in days (pulled up by the long tail).
- **`median_ae_days`** — median absolute error in days (robust to outliers).
- **`r2_log`** — R² on the `log(minutes)` scale the models are trained on.

### 9.2 The comparison

`results/model_comparison.txt`, on the same held-out 20% of the 1,134 resolved
issues (seed 0):

| Model | c_index | mae_days | median_ae_days | r2_log |
| --- | ---: | ---: | ---: | ---: |
| Linear regression (OLS) | 0.613 | 50.0 | 5.4 | 0.130 |
| Decision tree | 0.581 | 45.4 | 6.6 | 0.109 |
| Random forest | 0.636 | 48.7 | 5.2 | 0.196 |
| Cox PH | 0.597 | 56.8 | 12.0 | n/a |
| Random survival forest | 0.617 | 45.0 | 9.8 | n/a |

Censored-aware C-index over *all* 1,458 issues (80/20 split): Cox **0.619**,
RSF **0.625**.

### 9.3 Reading the table

- The **random forest** has the best R² on the log scale (≈ 0.20) and the best
  ranking of resolution times (C-index ≈ 0.64).
- The **random survival forest** achieves the lowest *mean* absolute error
  (~45 days) while also exploiting the censored issues.
- The **linear model** is the most interpretable but not the most accurate —
  exactly the trade-off this exercise highlights.
- All models share a median absolute error around 5–10 days, which sounds great
  but is deceptive: it mostly reflects that *most issues are short*. The long
  tail is systematically missed — see [Section 11](#11-discussion-why-the-predictions-look-too-small).

### 9.4 Per-family notes

- **Linear regression** — easy to interpret (coefficients with p-values in the
  OLS summary), but can behave badly on very long-running issues.
- **Decision tree** — beautifully readable (`tree.dot`), but overfits easily;
  it is the only model that occasionally predicts large values (up to ~580
  days), which is not necessarily a good thing.
- **Random forest** — the safer default for prediction; it averages the trees
  and caps out around ~100 days.
- **Cox PH** — interpretable *hazard ratios*: which features increase/decrease
  the risk of resolution over time.
- **Random survival forest** — best all-round survival model, and the only one
  that natively produces remaining-time forecasts and P(resolve in 30/90/180
  days) for the still-open issues.

---

## 10. Answers to the exercise

### 10.1 Which data should be used to train the model?

If you want to use a supervised learning technique, only issues with a known resolution date
can be used, because the closing time can only be
computed for those rows — that gives the **1,134 training issues**.
Otherwise, survival models allows to use also the remaining observation without a label,
treating them as censored observations.

### 10.2 Which variables should be kept?

Keep the variables that can realistically be known when a new issue arrives:

- `priority`, `issue_type`, `reporter`
- `description_length`, `summary_length`
- `issue_number` (era/trend proxy)
- `num_components`, `component_java`, `num_affected_versions`, `num_labels`
- `created_weekday`, `created_month`

Remove the variables that would leak the answer or are not physically available for a new
issue:

- `status`, `resolution`, `assignee` (leak or unavailable at creation)
- `created`, `updated`, `resolutiondate` (used to compute the target)
- `project` (constant), `key` (unique), `days_in_current_status`
- `vote_count`, `comment_count`, `watch_count` (circular — they grow with age)

In practice the linear model is built on the **7-feature final set** described
in [Section 8.5](#85-the-final-feature-set).

### 10.3 How were the categorical variables handled?

- Rare reporters were grouped into `Other`.
- `issue_type` was reduced to two groups: `Short` and `Long`.
- The final model was built on one-hot encoded variables (`drop_first=True`).

### 10.4 What do the models tell us?

The linear model gives a clear, interpretable baseline. The tree models behave
better on unusual, very old, or far-from-average issues. In short: linear for
interpretation, trees for prediction on messy real data, survival models for
the planning question ("will it still be open in three months?").

### 10.5 What happens on the interesting forecasting-pilot cases?

Two patterns stand out on the 324 still-open issues:

- Issues that have stayed open for a long time tend to receive long predicted
  resolution times.
- Issues that already have a patch available, or active community discussion,
  are usually predicted to close sooner.

That matches the intuition from the exploratory analysis: the age of the issue
and the level of activity around it both matter.

---

## 11. Discussion

Let's deep dive a second in one important aspect of the analysis: all the predictions 
from the proposed models look "very small" in general. Why is this so?

> The target is log-transformed

Resolution times are extremely right-skewed: most issues close in a few days
(**median ≈ 6 days**), but the **mean is ≈ 47 days** and the longest issue took
**≈ 952 days**. A small number of very long issues sit far from the majority.
Modeling the **logarithm** of the duration is the standard fix: without it, a
handful of year-long issues would dominate the fit and the rest of the data
would be ignored.

> The models predict the *typical*, not the *average*, case

A model trained to minimize squared error on `log(duration)` learns to predict
the **median** resolution time, not the mean. When predictions are transformed
back to days, they read as *"a typical issue like this takes about X days"* —
not *"on average an issue like this takes X days"*. Because the long issues are
rare (~6% of issues take more than 180 days), the typical duration is much
smaller than the average. That is exactly why the predicted-vs-actual plots
cluster around small numbers.

> The long tail is systematically missed

Looking only at issues that really took a long time:

- For issues that took **more than 90 days**, the median prediction of *every*
  model family is still only **about 7 days**.
- For almost all of those long issues (90–100%), the prediction is less than
  *half* of the real duration.

Three things reinforce each other:

1. **The objective targets the middle, not the tail.** Minimizing error on
   `log(duration)` optimizes for the typical case, so predictions land below
   the average by a factor of about 8 on the day scale.
2. **Regression to the middle.** The creation-time features are only moderately
   informative (the correlation on the log scale is ≈ 0.42). With little
   signal, the safest prediction is close to the overall typical duration
   (~6 days) — for every issue, including the long ones.
3. **The standard metrics reward this behavior.** MAE is pulled up by the few
   long issues, but the *median* absolute error looks excellent precisely
   because most issues are short. A model predicting "about 6 days" for
   everything scores well on the median yet never catches the interesting long
   cases.

In short: the models answer *"how long does a typical issue take?"* honestly
and reliably. They do **not** answer *"how long will this potentially
long-running issue take?"* — and a median-oriented estimator cannot
systematically produce long predictions given the signal available at creation
time.

The result is that **Very long issues are systematically under-predicted.** Because the target
is log-transformed and heavily skewed, the models optimize for the *typical*
resolution time and rarely produce large predictions (see
[Section 11](#11-discussion-why-the-predictions-look-too-small)). The models
are therefore not useful for spotting the rare, very long-running issues that
matter most in practice.

### 11.1 Where the survival models help

Survival analysis (scripts 8 and 9) is a partial answer to the long-tail
problem. Instead of a single "typical" number it models the whole resolution
*distribution* and can condition on how long an issue has already been open.
For the 324 still-open issues it reports P(resolve in 30/90/180 days) and a
remaining-time estimate — much closer to the planning question that really
matters: *"will this issue still be open in three months?"*

---

## 12. Future work

All natural candidates to improve the models, especially on the long tail:

- **Predict a high quantile** (e.g. the 90th percentile) of the duration
  instead of the median, so the model is explicitly trained to catch the long
  cases (quantile regression for the linear model, a quantile loss for the tree
  models).
- **Apply a bias correction ("smearing").** Multiplying the predictions by the
  average exponential residual (~8 here) fixes the overall level, although it
  does not improve the ranking.
- **Weight the training examples by their duration**, or use a two-step
  approach: first classify "will this take more than 30 days?", then predict the
  duration only inside the long group.
- **Add tail-aware evaluation metrics** (e.g. error only on the issues that
  really took long, or accuracy at the 90th percentile) so progress on the
  important cases is visible in the model comparison.

---

## 13. Appendix — key outputs, folder by folder

| Folder | Key files | What they are |
| --- | --- | --- |
| `data_sets/` | `raw/avro-issues.csv` | original flat CSV (never modified) |
| | `raw/avro-issues.json` | original nested JSON export |
| | `raw/avro-issues-merged.csv` | CSV + JSON features merged |
| | `dataset.csv`, `processed.csv`, `transformed_nonencoded.csv`, `encoded.csv` | the four pipeline datasets (see [§2.4](#24-data-flow-who-writes-what)) |
| | `forecasting_pilot.csv` | the 324 still-open issues |
| `exploratory_analysis/` | `datasetInfo.txt` | column-by-column report |
| | `freq_*.png`, `histogram.png`, `plotbox.png` | first distributions |
| `bivariate_analysis/` | `numerical/`, `full_classes/`, `reduced_classes/` | target-vs-predictor plots |
| | `duration_log.png` | log-transformed target |
| `feature_selection/` | `CVscoresVSfeatures_comparison.png` | the 4 techniques' CV curves |
| | `best_sequential.joblib` | winning technique's feature indices |
| | `sequential_subsets.joblib` | all 4 techniques' best subsets + scores |
| | `final_features.joblib` | the 7-feature final set used by the linear model |
| | `feature_selection_summary.txt` | Ridge/Lasso cross-check transcript |
| | `ridge_*.png`, `lasso_*.png` | regularization diagnostics |
| `multi_lin_reg/` | `model_metrics.txt`, `linreg.joblib`, `predictedVStest*.png`, `residuals/` | linear model results |
| `regression_tree/` | `model_metrics.txt`, `tree.joblib`, `rfr.joblib`, `tree.dot`, `ParametersImportance_rfr.png` | tree model results |
| `survival_analysis/` | `model_metrics.txt`, `km_overall.png`, `km_by_issue_type.png`, `cph.joblib`, `rsf.joblib`, `survival_eval.joblib` | survival results |
| `question2/` | `forecasting_pilot_predictions.csv`, `forecasting_pilot_results.txt`, `predictionComparison*.png` | forecasting pilot on open issues |
| `results/` | `model_comparison.txt` | the final all-model comparison |
| `about/` | `description_avro.pdf`, `ENEL_AVROcase.pdf`, `AvroCaseBragantini.pptx` | the original exercise description |
