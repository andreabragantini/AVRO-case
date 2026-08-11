# -*- coding: utf-8 -*-
"""
5_feature_selection.py - Sequential feature selection (educational comparison)
===============================================================================

ROLE IN THE PIPELINE
--------------------
This script sits right after the encoding step (4_encoding.py) and before the
model-design step (6_multi_lin_reg.py). It reduces the 32 encoded features to a
smaller, interpretable set using FOUR different *sequential* search techniques,
compares them, and persists the winning subset to disk.

WHY FEATURE SELECTION AT ALL?
-----------------------------
A linear model with all 32 features is still usable, but a smaller subset is
easier to interpret and can generalise better. Feature selection answers the
question: "which subset of the available columns actually carries signal?"

THE FOUR TECHNIQUES (all from the mlxtend package)
--------------------------------------------------
All four are *wrapper* methods: they score a candidate subset by actually
fitting a LinearRegression and measuring its cross-validation error. They only
differ in HOW they move through the space of subsets:

1. SFS_f - Sequential Forward Selection.
   Start with an empty set. At every step add the ONE feature that improves the
   CV score the most. Never removes anything afterwards. Greedy.

2. SBS - Sequential Backward Selection.
   Start with ALL features. At every step REMOVE the one feature whose removal
   hurts the CV score the least. Also greedy, but in the opposite direction.

3. SFFS - Sequential Floating Forward Selection.
   Like SFS_f, but AFTER each addition it may remove a feature again if that
   improves the score. The "floating" step lets it escape bad greedy choices.

4. SFBS - Sequential Floating Backward Selection.
   Like SBS, but after each removal it may add a feature back if that helps.

Floating variants search more of the space, so they usually find at least as
good a subset - at the cost of more computation.

CROSS-VALIDATION AND THE SCORING CONVENTION
-------------------------------------------
Every candidate subset is scored with 5-fold cross-validation (cv=5). The
mlxtend/sklearn convention is that HIGHER scores are always better, so the
mean-squared-error is returned as its negation, 'neg_mean_squared_error'
(i.e. a *negative* MSE). We undo the negation (multiply by -1) wherever we want
a plain "lower is better" MSE to read naturally.

WHY NO STANDARDISATION HERE?
----------------------------
LinearRegression solves a least-squares problem, and least squares is
*scale-invariant*: multiplying a feature by a constant does not change the
predictions or the R^2/MSE. So we can feed the raw encoded values straight in.
(Standardisation only matters for penalised models such as Ridge/Lasso, whose
penalty acts on the coefficients directly - see 5_ridge_lasso.py.)

WHY NOT AN EXHAUSTIVE SEARCH (EFS)?
-----------------------------------
An exhaustive search tries EVERY possible subset (2^32 ~ 4 billion here), which
is computationally impossible. That is why mlxtend's ExhaustiveFeatureSelector
is deliberately NOT used. Sequential searches walk a tiny fraction of that space
and are the standard practical compromise.

HOW THE WINNER IS CHOSEN (the selection rule)
---------------------------------------------
Each technique picks its own best feature count k (the one with the lowest mean
5-fold CV MSE). We then compare those four "best subsets" against each other.
Because every technique uses the SAME 5-fold split (sklearn's default KFold is
deterministic and unshuffled), the four mean CV scores are directly comparable,
so picking the lowest one is a fair, principled choice - unlike arbitrarily
favouring one technique.

OUTPUTS (feature_selection/)
    CVscoresVSfeatures_comparison.png   - the 4 CV-MSE-vs-#features curves
    sequential_subsets.joblib           - every technique's best subset + scores
    best_sequential.joblib              - indices of the overall winning subset
    (the legacy feature_selection/b1_features.joblib is NO LONGER written;
     its role is superseded by best_sequential.joblib)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

from utils.avro_common import print_section

print_section('5. FEATURE SELECTION (sequential searches)')

# ---------------------------------------------------------------------------
# 1. Load the encoded data
# ---------------------------------------------------------------------------
# encoded.csv holds the log-transformed numeric features, the one-hot encoded
# categoricals and the target in the LAST column. The target ("duration") is the
# resolution time in log(minutes); all other columns are candidate features.
data = pd.read_csv('data_sets/encoded.csv')
X = data.iloc[:, :-1]          # candidate features
y = data['duration']           # target
feature_names = list(X.columns)
n_features = X.shape[1]

# create the output directory (the runner also pre-creates it)
if not os.path.exists('feature_selection'):
    os.makedirs('feature_selection')

# ---------------------------------------------------------------------------
# 2. Configuration of the four techniques
# ---------------------------------------------------------------------------
# Each entry is (label, forward, floating):
#   forward  = add features one by one (True)  or remove them one by one (False)
#   floating = allow undo moves (True) or never undo (False)
TECHNIQUES = [
    ('SFS_f', True,  False),   # Sequential Forward Selection
    ('SBS',   False, False),   # Sequential Backward Selection
    ('SFFS',  True,  True),    # Sequential Floating Forward Selection
    ('SFBS',  False, True),    # Sequential Floating Backward Selection
]


def run_sequential(forward, floating, label):
    """Run one sequential search and summarise its results.

    Parameters
    ----------
    forward : bool
        True for forward search (add features), False for backward (remove).
    floating : bool
        True to allow conditional add/remove "undo" moves.
    label : str
        Human-readable name, used only for messages and the plot.

    Returns
    -------
    ks : list of int
        Every feature count that was evaluated (1 .. n_features). These are
        also the keys of the metric dict, which is why we sort them instead of
        assuming the position of the best one (see note below).
    mean_cv : list of float
        Mean 5-fold CV MSE for each count in `ks` (lower is better).
    best_k : int
        Feature count with the lowest mean CV MSE.
    best_idx : list of int
        Column indices of the best subset (used to persist the selection).
    best_mse : float
        Mean CV MSE of the best subset.

    Note about the "off-by-one" trap
    --------------------------------
    The metric dict returned by mlxtend is keyed by the NUMBER of features in
    that step (1, 2, ..., n_features). If we stored the MSE series in a plain
    list, position 0 would correspond to 1 feature, position i to i+1 features,
    and naively using np.argmin(...) as a dict key would silently grab the WRONG
    subset (one feature short). Sorting the keys and working with the actual
    feature count avoids that class of bug entirely.
    """
    sfs = SFS(
        LinearRegression(),          # estimator used to score every subset
        k_features=(1, n_features),  # explore every count from 1 to all features
        forward=forward,
        floating=floating,
        scoring='neg_mean_squared_error',  # higher-better convention (negated)
        cv=5,                              # identical 5-fold split for all runs
        n_jobs=-1,                         # use all CPU cores to speed this up
    )
    sfs = sfs.fit(X.values, y.values)
    metric = sfs.get_metric_dict()   # {feature_count: {'cv_scores': [...], ...}}

    # Reconstruct the "mean CV MSE vs #features" curve, undoing the negation
    # so a lower number means a better (smaller) error.
    ks = sorted(metric.keys())
    mean_cv = [-np.mean(metric[k]['cv_scores']) for k in ks]

    # Best count = the one with the smallest mean CV MSE.
    best_pos = int(np.argmin(mean_cv))
    best_k = ks[best_pos]
    best_mse = mean_cv[best_pos]
    best_idx = list(metric[best_k]['feature_idx'])

    return ks, mean_cv, best_k, best_idx, best_mse


def choose_winner(results):
    """Pick the best technique.

    The rule: lowest best-k mean CV MSE wins. If two techniques are
    indistinguishable (within a tiny tolerance), prefer the one using fewer
    features - the simpler model is preferred when everything else is equal.
    """
    best_label = None
    best_mse = np.inf
    best_k = np.inf
    for label, res in results.items():
        better_score = res['best_mse'] < best_mse - 1e-12
        tie_break = abs(res['best_mse'] - best_mse) <= 1e-12 and res['best_k'] < best_k
        if better_score or tie_break:
            best_label = label
            best_mse = res['best_mse']
            best_k = res['best_k']
    return best_label

# ---------------------------------------------------------------------------
# 3. Run all four techniques
# ---------------------------------------------------------------------------
# results[label] holds everything we need: the curve (ks, mean_cv), the chosen
# subset (best_idx) and its score (best_k, best_mse).
results = {}
plt.figure(figsize=(12, 8))
for label, forward, floating in TECHNIQUES:
    print_section('Sequential {} ({})'.format(
        'Forward Selection' if forward else 'Backward Selection', label))

    ks, mean_cv, best_k, best_idx, best_mse = run_sequential(
        forward, floating, label)
    results[label] = {
        'ks': ks, 'mean_cv': mean_cv,
        'best_k': best_k, 'best_idx': best_idx, 'best_mse': best_mse,
    }

    # One curve per technique on the shared comparison plot.
    plt.plot(ks, mean_cv, marker='.', label='{} (best k={}, MSE={:.4f})'.format(
        label, best_k, best_mse))

    print('Best number of features: {}'.format(best_k))
    print('Best mean CV MSE: {:.4f}'.format(best_mse))
    print('Selected columns ({}): {}'.format(len(best_idx),
                                             ', '.join(feature_names[i] for i in best_idx)))

# Finish the comparison plot (legend, axes, grid, title).
plt.scatter([results[l]['best_k'] for l in results],
            [results[l]['best_mse'] for l in results],
            marker='o', s=90, facecolors='none', edgecolors='red',
            label='best count per technique')
plt.xlabel('Number of features')
plt.ylabel('Mean CV MSE')
plt.title('Sequential feature selection: mean CV MSE vs number of features')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('feature_selection/CVscoresVSfeatures_comparison.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 4. Final comparison table + principled winner
# ---------------------------------------------------------------------------
print_section('Final feature selection summary')

summary = pd.DataFrame([
    {
        'technique': label,
        'best_k': res['best_k'],
        'best_mse': res['best_mse'],
        'features': len(res['best_idx']),
    }
    for label, res in results.items()
]).sort_values('best_mse').reset_index(drop=True)
print('\nPer-technique results (sorted by mean CV MSE, lower is better):')
print(summary.to_string(index=False))

winner = choose_winner(results)
winner_idx = results[winner]['best_idx']
print('\n>>> Winner: {} with {} features and mean CV MSE {:.4f}'.format(
    winner, len(winner_idx), results[winner]['best_mse']))
print('    Winning feature set: {}'.format(
    ', '.join(feature_names[i] for i in winner_idx)))

# Which features are selected by each technique (1 = selected, 0 = not).
all_selected = sorted({
    name
    for res in results.values()
    for name in (feature_names[i] for i in res['best_idx'])
})
selection_matrix = pd.DataFrame(
    {
        label: [1 if name in {feature_names[i] for i in res['best_idx']} else 0
                for name in all_selected]
        for label, res in results.items()
    },
    index=all_selected,
)
print('\nWhich features are selected by each technique (1 = selected):')
print(selection_matrix)

# ---------------------------------------------------------------------------
# 5. Persist the results so the later scripts can run standalone
# ---------------------------------------------------------------------------
# Every technique's best subset plus its score, for later reference and for
# 5_ridge_lasso.py to know which technique won.
dump(
    {
        label: {
            'feature_idx': res['best_idx'],
            'k': res['best_k'],
            'mse': res['best_mse'],
        }
        for label, res in results.items()
    },
    'feature_selection/sequential_subsets.joblib',
)
print('\nSaved per-technique best subsets -> feature_selection/sequential_subsets.joblib')

# The overall winner as a plain list of indices - the canonical "sequential
# selection result" consumed by 5_ridge_lasso.py and (as a fallback) by
# 6_multi_lin_reg.py.
dump(winner_idx, 'feature_selection/best_sequential.joblib')
print('Saved winning feature indices -> feature_selection/best_sequential.joblib')
