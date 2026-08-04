# AVRO-case

## Introduction

This repository contains a **data science exercise** built around the public
issue tracker of **Apache Avro**, an open-source data serialization framework
(a bit like JSON or Protobuf) used mainly in big-data systems. The Apache
Avro project manages its bug reports, feature requests and other tasks as
"issues" in a JIRA tracker, just like many other open-source projects.

Every issue comes with a set of details: who reported it, its priority, its
type (bug, feature request, ...), how many people voted for or commented on
it, and when it was created. If the issue has already been resolved, the
tracker also knows *when* it was resolved.

The question this exercise tries to answer is a practical one:

> Given the information available when a new issue is opened, can we predict
> **how long it will take to resolve it**?

Knowing the expected resolution time in advance would help a project plan its
work, allocate contributors, and set expectations with the community.

The repository uses a snapshot of **1,458 issues** from the Apache Avro JIRA
tracker, explores the data, and trains several machine-learning models — a
simple linear regression, Ridge/Lasso, and tree-based models — to predict the
resolution time. The focus is on **simple, explainable models** that can be
inspected and understood, rather than on squeezing out the last bit of
accuracy. The rest of this document describes the dataset, the preprocessing
choices, the models that were tried, and the main findings.

## What This Study Does

The raw dataset contains 1,458 issues. Not all of them can be used to train a
prediction model, because only issues with a valid creation date and resolution
date can tell us the real closing time.

After preprocessing:

- 1,134 issues (those with a valid resolution date) are used for training.
- 324 issues were still open at the time of the snapshot. They have no
  resolution date and therefore no ground truth. 
  They are kept as a *forecasting pilot*: a qualitative sanity
  check on realistic, still-open cases (see `9_predicting.py`). Quantitative
  evaluation is done on a random 80/20 train/test split of the 1,134 resolved
  issues inside the modeling scripts.
- The target variable is the resolution time, measured from `created` to
  `resolutiondate`.

The analysis focuses on simple, explainable models first and then compares them
with tree-based models.

## Main Findings

- Most issues are solved in a short time, but the distribution is very skewed.
  A few issues stay open for months or years and behave like strong outliers.
- Some fields would leak the answer or are not available for a future issue, so
  they are removed from the model. The most important ones are `status`,
  `resolution`, `assignee`, and the date fields used to compute the target.
- The most useful signals are `comment_count`, `watch_count`, `vote_count`,
  `issue_type`, and a few frequent reporters.
- `issue_type` matters more after grouping the classes into two simple groups:
  `Short` and `Long`.
- The linear model reaches an R-squared of about `0.336` on the training fit in
  the saved OLS summary, so it explains part of the variance but not all of it.
- The linear model is useful for interpretation, but it can behave badly on very
  long-running issues. Tree models are more robust on those extreme cases.
- A single decision tree overfits easily. The random forest is the safer choice
  when the goal is to predict, not just explain.

## Answers To The Exercise

### 1. Which data should be used to train the model?

Only issues with a known resolution date should be used for training, because
the closing time can only be computed for those rows. That gives the 1,134
training issues.

### 2. Which variables should be kept?

Keep the variables that can realistically be known when a new issue arrives:

- `priority`
- `issue_type`
- `reporter`
- `vote_count`
- `comment_count`
- `description_length`
- `summary_length`
- `watch_count`

Remove the variables that would not be available for a new issue or would leak
the answer:

- `status`
- `resolution`
- `assignee`
- `created`
- `updated`
- `resolutiondate`
- `project`
- `key`
- `days_in_current_status`

### 3. How were the categorical variables handled?

The categories were simplified before encoding:

- Rare reporters were grouped into `Other`.
- `issue_type` was reduced to two groups: `Short` and `Long`.
- The final model was built on one-hot encoded variables.

### 4. What do the models tell us?

The linear model gives a clear baseline and makes it easy to see which features
matter most. The tree models are better when the issue is unusual, very old, or
far from the average case. In practice, that means the linear model is good for
interpretation, while the tree models are better for prediction on messy real
data.

### 5. What happens on the interesting validation cases?

The validation examples show two patterns:

- Issues that have stayed open for a long time tend to receive long predicted
  resolution times.
- Issues that already have a patch available, or that have active community
  discussion, are usually predicted to close sooner.

That matches the intuition from the exploratory analysis: the age of the issue
and the level of activity around it both matter.

## Quick Start

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Then run the full analysis from the repository root:

```bash
python run_analysis.py
```

This runs the scripts in one shared Python session so later stages can reuse
the models and variables created earlier in the workflow. The runner also uses a
non-interactive Matplotlib backend, so plots are saved to disk instead of
opening windows during execution.

## Pipeline Order

The main end-to-end flow is:

1. `1_exploratory_analysis.py`
2. `2_preprocessing.py`
3. `3_bivariate_analysis.py`
4. `4_encoding.py`
5. `5_feature_selection.py`
6. `ridge_lasso.py`
7. `6_multi_lin_reg.py`
8. `7_regression_trees.py`
9. `9_predicting.py`

## Known Limitations

The baseline analysis below has two documented limitations:

- **`issue_type` grouping is mildly target-informed.** The decision to group
  issue types into `Short` and `Long` was taken in the bivariate analysis
  after observing the resolution times per type. This is a simplification that
  is kept for clarity, but it means the target has influenced how the feature
  was built.
- **Activity counts are partly circular.** `comment_count`, `vote_count` and
  `watch_count` grow with the age of an issue, so they correlate with the
  resolution time partly *because* a long-lived issue has simply had more time
  to accumulate activity. This inflates their apparent predictive power and
  makes them imperfect for forecasting a brand-new issue, where those counts
  are not yet known.

Both caveats are acknowledged rather than hidden; a more rigorous treatment
would require a fixed-snapshot definition of the count features and raw
`issue_type` classes.

## Notes

- `ridge_lasso.py` is part of the main analysis, not a separate side project.
  It sits after feature selection as a regularization check and helps compare
  Ridge and Lasso against the encoded feature set used by the linear models.
- `8exploreJSON.py` is a standalone inspection helper for the raw JSON source.
  It is useful for data exploration, but it is not part of the main pipeline.
- `9_predicting.py` produces the forecasting-pilot comparison figures for the
  still-open issues in `question2/`.
