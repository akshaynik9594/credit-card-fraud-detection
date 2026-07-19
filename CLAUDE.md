# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A cost-sensitive, explainable credit-card fraud detection pipeline. It deliberately optimizes for business cost rather than accuracy, since fraud is ~0.17% of transactions (a "never fraud" model would be 99.8% accurate and useless). Two models are trained (logistic regression baseline, XGBoost challenger), evaluated on PR-AUC, and the decision threshold is chosen to minimize total expected cost (FP = £5, FN = £500) rather than defaulting to 0.5. SHAP provides global and per-transaction explainability.

## Commands

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib

# Only needed if data/creditcard.csv (real ULB dataset) is not present
python src/make_synthetic_data.py

# Runs the full pipeline: trains both models, tunes the cost threshold,
# generates SHAP plots. Prints PR-AUC/ROC-AUC and cost comparison to stdout.
python src/train.py
```

There is no test suite, linter, or build step in this repo — it's a single training script.

## Architecture

- `src/train.py` is the entire pipeline, run top-to-bottom as a script. The model-train + cost-threshold logic is factored into one function, `evaluate_split(X_tr, X_te, y_tr, y_te, label)`, which is called twice — once per evaluation protocol — rather than being organized into further modules. It does, in order:
  1. Loads `data/creditcard.csv` if present, else falls back to `data/creditcard_synthetic.csv` (path selection logic at the top of the file).
  2. Builds two train/test splits: a stratified **random split** (`train_test_split`, the classic protocol) and a **time-based split** (sort by `Time`, train on the first 70% chronologically, test on the last 30% — simulates a model that only ever sees the past, unlike the random split which lets future transactions leak into training).
  3. Runs `evaluate_split` on each: scales `Time`/`Amount` for the logistic regression baseline only (XGBoost trains on unscaled features), trains logistic regression (`class_weight="balanced"`) and XGBoost (`scale_pos_weight` set from that split's train-set class ratio), then sweeps 197 thresholds to find the one minimizing `fp * COST_FP + fn * COST_FN` (constants at top: `COST_FP = 5`, `COST_FN = 500`) versus the naive 0.5 threshold.
  4. Prints a side-by-side comparison table (PR-AUC, ROC-AUC, cost-optimal threshold, optimal vs. naive cost) for the two splits.
  5. Generates four charts into `outputs/`: a 3-panel random-vs-time comparison (`04_split_comparison.png`: PR curve overlay, cost-vs-threshold overlay, naive-vs-optimal cost bars), the primary PR curve + cost-vs-threshold chart for the random split (`01_pr_curve_and_cost.png`), a global SHAP summary (`02_shap_summary.png`), and a waterfall explanation for the single highest-risk test transaction (`03_shap_single_case.png`).
  - SHAP explainability runs only on the **random-split** XGBoost model (`primary = random_res`) — it is not duplicated for the time split.
  - Changing the cost assumptions or model hyperparameters means editing the constants/model calls directly in this file.

- `src/make_synthetic_data.py` generates a schema-matched fallback dataset (`Time`, `V1..V28` PCA-like components, `Amount`, `Class`) with the same ~0.17% fraud rate, so the pipeline can run end-to-end without downloading the real dataset. It's a synthetic approximation only — real results/publication should use the actual ULB dataset.

- `data/` holds `creditcard.csv` (real dataset, gitignored, must be downloaded manually from the [ULB Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) and/or `creditcard_synthetic.csv` (generated, gitignored).

- `outputs/` holds the four generated PNGs, overwritten each run of `train.py`.

## Key design decisions to preserve

- **PR-AUC, not accuracy or ROC-AUC**, is the headline metric — ROC-AUC looks nearly identical between models under this extreme imbalance and hides the real performance gap.
- **Threshold is chosen by cost minimization**, not fixed at 0.5 — any change to evaluation should keep threshold selection driven by `total_cost()`, not a hardcoded cutoff.
- **Both a random split and a time-based split are evaluated and compared**, not just one — on the real dataset the time split shows lower PR-AUC but higher ROC-AUC than the random split, illustrating why the random-split protocol alone can overstate real-world performance. Any new evaluation code should go through `evaluate_split()` so it stays available under both protocols rather than only the random one.
