# -*- coding: utf-8 -*-
"""
5_ridge_lasso.py - Regularization as a cross-check on feature selection
========================================================================

ROLE IN THE PIPELINE
--------------------
This script sits right after the encoding step (4_encoding.py) and before the
model-design step (6_multi_lin_reg.py). It runs Ridge (L2) and Lasso (L1)
regression on the encoded training set.

IMPORTANT - WHAT THIS SCRIPT IS (AND IS NOT)
--------------------------------------------
Ridge and Lasso are regularized linear models. Regularization is classically
used to fight OVERFITTING. In this project that is NOT the main concern: the
plain linear model barely overfits (train/test R^2 gap ~0.09 with ~35x more
rows than features), so shrinkage buys little in predictive terms.

The real value of Ridge/Lasso here is FEATURE SELECTION CROSS-CHECK:
  - Lasso (L1 penalty) can shrink coefficients EXACTLY to zero, so it is an
    independent, automatic feature selector. The features it keeps should be
    compared against the sequential-selection result saved by the
    feature-selection step (feature_selection/best_sequential.joblib, the
    winner among the four sequential techniques of 5_feature_selection.py).
  - Ridge (L2 penalty) never zeroes coefficients but tells us which features
    are consistently strong (largest |coefficient|) as the penalty grows.

For this reason this script does NOT write a model-metrics table like the
predictive-model scripts (6/7/8): Ridge and Lasso are used as selection
diagnostics, not as candidate predictive models.

WHAT THIS SCRIPT DOES
---------------------
1. Scale the features (StandardScaler - required so that the L1/L2 penalty
   treats every feature fairly regardless of its original units).
2. Choose the penalty strength alpha by CROSS-VALIDATION (RidgeCV/LassoCV)
   instead of a hand-picked value.
3. Print a one-line R^2 sanity check: the Lasso-selected subset must still
   explain the target reasonably well.
4. Compare the features kept by Lasso against the best sequential-selection
   features (winner of 5_feature_selection.py).
5. Report the strongest Ridge features (by |coefficient|) as a stability check.
6. Cross-check the candidate feature sets by 5-fold CV R^2 and persist the
   winner to feature_selection/final_features.joblib, which is the single
   feature set 6_multi_lin_reg.py builds the linear model on. The candidates
   are the best sequential set, the Lasso set, their overlap, and all features.
7. Show how the number of features Lasso keeps changes with alpha (sparsity).
8. Save two diagnostic plots per model (alpha-vs-R^2 and coefficient paths).
9. Tee the whole console output into feature_selection_summary.txt so the
   comparison survives beyond the terminal.

OUTPUTS (feature_selection/)
    ridge_train_test_r2_vs_alpha.png
    ridge_coefficient_paths.png
    lasso_train_test_r2_vs_alpha.png
    lasso_coefficient_paths.png
    feature_selection_summary.txt   (full console transcript of the check)
    final_features.joblib           (CV-chosen feature indices for 6_multi_lin_reg.py)

NB: the alpha chosen by cross-validation is the
*data-driven* answer to "how strongly should we penalize?"; the feature
comparison is the *science* of this script - it confirms or contradicts the
greedy SFS selection using a completely different mathematical route.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import linear_model
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.avro_common import print_section

print_section('REGULARIZATION CHECK (RIDGE / LASSO) - FEATURE SELECTION CROSS-CHECK')

# ---------------------------------------------------------------------------
# 0. Load the encoded training data
# ---------------------------------------------------------------------------
# encoded.csv holds the log-transformed features + the log(minutes) target.
# The last column is the target variable ("duration"); all others are features.
data = pd.read_csv("data_sets/encoded.csv")
X = data.iloc[:, :-1]
y = data["duration"]
feature_names = list(X.columns)
n_features = X.shape[1]

if not os.path.exists("feature_selection"):
    os.makedirs("feature_selection")

# ---------------------------------------------------------------------------
# Console tee -> feature_selection/feature_selection_summary.txt
# ---------------------------------------------------------------------------
# Everything printed below is duplicated into the summary file so the
# Lasso-vs-SFS agreement, the Ridge ranking and the CV cross-check survive
# beyond the terminal. stdout is restored at the end of the script.
class _Tee:
    """Write every line to several streams at once (console + summary file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


_summary_path = os.path.join("feature_selection", "feature_selection_summary.txt")
_summary_stream = open(_summary_path, "w")
sys.stdout = _Tee(sys.__stdout__, _summary_stream)

# ---------------------------------------------------------------------------
# 1. Train/test split + scaling
# ---------------------------------------------------------------------------
# WHY SCALING? The penalty term is applied to the COEFFICIENTS directly. If
# features keep their original units (e.g. description_length in the thousands
# vs one-hot 0/1 columns), the penalty would shrink large-range features
# disproportionately and the result would reflect feature SCALE, not feature
# IMPORTANCE. StandardScaler (zero mean, unit variance) fixes that. It is fit
# on the training split ONLY to avoid leaking test-set statistics.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Cross-validated alpha selection
# ---------------------------------------------------------------------------
# alpha is the penalty strength. Instead of guessing, we let 5-fold CV pick the
# value that maximizes R^2 on the training folds (RidgeCV / LassoCV).
# The grids are log-spaced to explore several orders of magnitude cheaply.
print_section('Ridge regression (L2) - cross-validated alpha')

ridge_alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=5).fit(X_train_scaled, y_train)
print('CV-selected alpha: {:.4f}'.format(ridge_cv.alpha_))
# One-line sanity check only (this script is a diagnostic, not a model).
print('Ridge R^2 on test set (log-minutes): {:.3f}'.format(
    ridge_cv.score(X_test_scaled, y_test)))

print_section('Lasso regression (L1) - cross-validated alpha')

lasso_alphas = np.logspace(-4, 1, 50)
lasso_cv = LassoCV(
    alphas=lasso_alphas, cv=5, max_iter=100000, random_state=0
).fit(X_train_scaled, y_train)
print('CV-selected alpha: {:.4f}'.format(lasso_cv.alpha_))
print('Lasso R^2 on test set (log-minutes): {:.3f}'.format(
    lasso_cv.score(X_test_scaled, y_test)))

# ---------------------------------------------------------------------------
# 3. Lasso as an automatic feature selector
# ---------------------------------------------------------------------------
# At the CV-selected alpha, Lasso keeps only the features it considers useful.
lasso_selected = [
    name for name, coef in zip(feature_names, lasso_cv.coef_)
    if abs(coef) > 1e-10
]
print('\nFeatures kept by Lasso at alpha={:.4f} ({} of {}):'.format(
    lasso_cv.alpha_, len(lasso_selected), n_features))
print('  ' + ', '.join(lasso_selected))

# ---------------------------------------------------------------------------
# 4. Compare Lasso's selection against the sequential-selection winner
# ---------------------------------------------------------------------------
# 5_feature_selection.py runs four sequential searches, picks the best one by
# mean CV MSE and saves its indices in feature_selection/best_sequential.joblib.
# Load them (if present) and compare with Lasso. The name of the winning
# technique is read from sequential_subsets.joblib for a nicer message.
print_section('Lasso vs best sequential feature set - feature agreement')

seq_selected = set()          # feature names chosen by the sequential winner
seq_desc = 'sequential selection'

try:
    seq_indices = load('feature_selection/best_sequential.joblib')
    seq_selected = {feature_names[i] for i in seq_indices}
    try:
        subsets = load('feature_selection/sequential_subsets.joblib')
        winner_label = min(subsets, key=lambda label: subsets[label]['mse'])
        seq_desc = '{} (winner of 5_feature_selection.py)'.format(winner_label)
    except (FileNotFoundError, KeyError):
        seq_desc = 'best sequential set'
    print('{} ({} of {}):'.format(seq_desc, len(seq_selected), n_features))
    print('  ' + ', '.join(sorted(seq_selected)))
except (FileNotFoundError, KeyError):
    # Legacy fallback: the old 5_feature_selection.py saved the plain SFS_f set
    # under this name. Keep reading it so older artifacts still work.
    try:
        seq_indices = load('feature_selection/b1_features.joblib')
        seq_selected = {feature_names[i] for i in seq_indices}
        seq_desc = 'legacy SFS set'
        print('{} ({} of {}):'.format(seq_desc, len(seq_selected), n_features))
        print('  ' + ', '.join(sorted(seq_selected)))
    except (FileNotFoundError, KeyError):
        print('feature_selection/best_sequential.joblib not found - comparison skipped.')

lasso_selected_set = set(lasso_selected)
in_both = set()
if seq_selected:
    in_both = lasso_selected_set & seq_selected
    lasso_only = lasso_selected_set - seq_selected
    seq_only = seq_selected - lasso_selected_set
    print('\nAgreement (kept by BOTH Lasso and {}): {} of {}'.format(
        seq_desc, len(in_both), len(seq_selected)))
    print('  ' + ', '.join(sorted(in_both)))
    print('\nKept ONLY by Lasso: {}'.format(len(lasso_only)))
    print('  ' + ', '.join(sorted(lasso_only)))
    print('\nKept ONLY by {}: {}'.format(seq_desc, len(seq_only)))
    print('  ' + ', '.join(sorted(seq_only)))

# ---------------------------------------------------------------------------
# 5. Ridge as a stability check (strongest features by |coefficient|)
# ---------------------------------------------------------------------------
# Ridge never zeroes coefficients, so report the features with the largest
# absolute standardized coefficient - the ones that matter most to the model.
print_section('Ridge strongest features (by |standardized coefficient|)')

ridge_strength = sorted(
    zip(feature_names, ridge_cv.coef_), key=lambda item: -abs(item[1])
)
print('Top {} features by |coefficient| (Ridge, alpha={:.4f}):'.format(
    min(10, n_features), ridge_cv.alpha_))
for name, coef in ridge_strength[:10]:
    print('  {:>6.3f}  {}'.format(coef, name))

# ---------------------------------------------------------------------------
# 6. Cross-check candidate feature sets by CV R^2 -> final feature set
# ---------------------------------------------------------------------------
# 6_multi_lin_reg.py needs ONE feature set for the interpretable linear model.
# Instead of trusting a single selection technique, compare the candidates by
# 5-fold CV R^2 on the scaled training data and persist the winner to
# feature_selection/final_features.joblib. The candidates are the best
# sequential set (winner of 5_feature_selection.py), the Lasso set, their
# overlap (the features both approaches agree on) and the full 32-feature set.
print_section('Final feature set: 5-fold CV R^2 cross-check')

from sklearn.model_selection import cross_val_score

# Build candidate feature sets
candidates = {}
if seq_selected:
    candidates['Best sequential'] = sorted(
        feature_names.index(name) for name in seq_selected)
candidates['Lasso'] = sorted(
    feature_names.index(name) for name in lasso_selected_set)
if in_both:
    candidates['Lasso & sequential overlap'] = sorted(
        feature_names.index(name) for name in in_both)
candidates['All features'] = list(range(n_features))

# Cross-validate each candidate and collect the mean/std of the 5-fold R^2.
cv_rows = []
for name, idx in candidates.items():
    scores = cross_val_score(
        LinearRegression(), X_train_scaled[:, idx], y_train, cv=5)
    cv_rows.append([name, len(idx), scores.mean(), scores.std()])

# Make it a DataFrame for pretty printing and sort by mean R^2.
cv_table = pd.DataFrame(
    cv_rows, columns=['feature set', 'n_features', 'cv_R2_mean', 'cv_R2_std'])
cv_table = cv_table.sort_values('cv_R2_mean', ascending=False).reset_index(drop=True)
print(cv_table.to_string(index=False))

# Persist the winning feature set to feature_selection/final_features.joblib
best_name = cv_table.iloc[0]['feature set']
best_idx = candidates[best_name]
print('\nWinning feature set: {} ({:.3f} CV R^2, {} features)'.format(
    best_name, cv_table.iloc[0]['cv_R2_mean'], len(best_idx)))
dump(best_idx, 'feature_selection/final_features.joblib')
print('Saved winning feature indices -> feature_selection/final_features.joblib')

# ---------------------------------------------------------------------------
# 7. Sparsity vs alpha (how many features Lasso keeps as penalty grows)
# ---------------------------------------------------------------------------
print_section('Lasso sparsity vs alpha')

sparsity_alphas = np.logspace(-4, 0, 25)
sparsity_counts = []
for alpha in sparsity_alphas:
    lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_train_scaled, y_train)
    sparsity_counts.append(np.sum(np.abs(lasso.coef_) > 1e-10))
sparsity = pd.DataFrame(
    {'alpha': sparsity_alphas, 'n_features_kept': sparsity_counts}
)
print(sparsity.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Diagnostic plots
# ---------------------------------------------------------------------------
# 8a. Alpha vs R^2 (Ridge and Lasso) - shows where the CV-selected alpha sits.
def plot_r2_vs_alpha(model, alphas, title, path, best_alpha):
    """Plot train/test R^2 over an alpha grid and mark the CV-selected alpha."""
    train_r2, test_r2 = [], []
    for alpha in alphas:
        fitted = model(alpha=alpha).fit(X_train_scaled, y_train)
        train_r2.append(fitted.score(X_train_scaled, y_train))
        test_r2.append(fitted.score(X_test_scaled, y_test))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(alphas, train_r2, marker='.', label='training R^2')
    ax.plot(alphas, test_r2, marker='.', label='test R^2')
    ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.8,
               label='CV-selected alpha = {:.4g}'.format(best_alpha))
    ax.set_xscale('log')
    ax.set_xlabel('alpha (penalty strength)')
    ax.set_ylabel('R^2')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


plot_r2_vs_alpha(
    Ridge, ridge_alphas,
    'Ridge: R^2 vs alpha (L2 regularization)',
    'feature_selection/ridge_train_test_r2_vs_alpha.png',
    ridge_cv.alpha_,
)
plot_r2_vs_alpha(
    Lasso, lasso_alphas,
    'Lasso: R^2 vs alpha (L1 regularization)',
    'feature_selection/lasso_train_test_r2_vs_alpha.png',
    lasso_cv.alpha_,
)

# 8b. Coefficient paths - how each feature's weight changes with alpha.
def plot_coefficient_path(alphas, coefs, title, path, best_alpha):
    """Plot coefficient shrinkage across alphas; mark the CV-selected alpha.

    coefs has shape (n_alphas, n_features): one row of weights per alpha.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(alphas, coefs)
    ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.8,
               label='CV-selected alpha = {:.4g}'.format(best_alpha))
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse so alpha increases left -> right
    ax.set_xlabel('alpha (penalty strength)')
    ax.set_ylabel('coefficient weight')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# Ridge path: fit Ridge across a fine alpha grid, collecting all coefficients.
ridge_path_alphas = np.logspace(-3, 3, 100)
ridge_path_coefs = np.array([
    linear_model.Ridge(alpha=a, fit_intercept=False).fit(
        X_train_scaled, y_train).coef_
    for a in ridge_path_alphas
])
plot_coefficient_path(
    ridge_path_alphas, ridge_path_coefs,
    'Ridge: coefficient shrinkage as alpha increases',
    'feature_selection/ridge_coefficient_paths.png',
    ridge_cv.alpha_,
)

# Lasso path: the LARS algorithm computes the full path in one pass.
print("Computing Lasso regularization path ...")
lasso_path_alphas, _, lasso_path_coefs = linear_model.lars_path(
    X_train_scaled, y_train.to_numpy(), method='lasso', verbose=False)
xx = np.sum(np.abs(lasso_path_coefs.T), axis=1)
xx /= xx[-1]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(xx, lasso_path_coefs.T, marker='.')
ymin, ymax = ax.get_ylim()
ax.vlines(xx, ymin, ymax, linestyle='dashed', alpha=0.3)
ax.set_xlabel('|coef| / max|coef| (L1 norm of the coefficient vector)')
ax.set_ylabel('Coefficients')
ax.set_title('Lasso path - coefficient shrinkage vs L1 norm')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('feature_selection/lasso_coefficient_paths.png', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Close the summary tee and restore the console
# ---------------------------------------------------------------------------
print('\nSaved 4 diagnostic plots to feature_selection/.')
print('\nSaved feature-selection summary -> {}'.format(_summary_path))
sys.stdout.flush()
sys.stdout = sys.__stdout__
_summary_stream.close()
