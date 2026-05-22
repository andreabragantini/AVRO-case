# AVRO-case

Data analysis and machine learning workflow for the AVRO case study.

## Quick start

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Then run the full analysis from the repository root:

```bash
python run_analysis.py
```

This runs the scripts in one shared Python session so the later stages can reuse
the models and variables created earlier in the workflow. The runner also uses a
non-interactive Matplotlib backend, so plots are saved to disk instead of opening
windows during execution.

## Pipeline order

The main end-to-end flow is:

1. `1exploratoryAnalysis.py`
2. `2preprocessing.py`
3. `3bivariateAnalysis.py`
4. `4encoding.py`
5. `5featureselection.py`
6. `ridge_lasso.py`
7. `6modeldesign.py`
8. `7regressiontrees.py`
9. `9predicting.py`

## Notes

- `ridge_lasso.py` is part of the main analysis, not a separate side project.
  It sits after feature selection as a regularization sanity check and helps
  compare Ridge and Lasso against the encoded feature set used by the linear
  models.
- `8exploreJSON.py` is a standalone inspection helper for the raw JSON source.
  It is useful for data exploration, but it is not part of the main pipeline.
- `9predicting.py` produces the validation-set comparison figures in
  `Question2/`.
