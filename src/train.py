"""
Credit Card Fraud Detection — cost-sensitive, explainable pipeline
-------------------------------------------------------------------
1. Baseline: logistic regression (class-weighted)
2. Challenger: XGBoost (scale_pos_weight for imbalance)
3. Evaluation that respects extreme imbalance: PR-AUC, not accuracy
4. Business-cost threshold tuning:
     false positive  -> £5   (ops review + customer friction)
     false negative  -> £500 (average fraud loss written off)
5. Two evaluation protocols compared side by side:
     - random split   (stratified 70/30, the classic protocol)
     - time-based split (sorted by Time, train on first 70%, test on last 30%,
       simulating deployment where the model only ever sees the past)
6. SHAP explainability: global drivers + one flagged transaction explained

Uses data/creditcard.csv (real ULB dataset) if present,
otherwise data/creditcard_synthetic.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, confusion_matrix)

COST_FP, COST_FN = 5, 500
OUT = "outputs"

# ---------- load ----------
path = "data/creditcard.csv" if os.path.exists("data/creditcard.csv") else "data/creditcard_synthetic.csv"
print(f"Using dataset: {path}" + ("  (REAL ULB data)" if "synthetic" not in path else "  (synthetic fallback — download the real set for publication)"))
df = pd.read_csv(path)
X = df.drop(columns=["Class"])
y = df["Class"]
print(f"{len(df):,} transactions | fraud rate {y.mean()*100:.3f}%")


def total_cost(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return fp * COST_FP + fn * COST_FN, fp, fn


def evaluate_split(X_tr, X_te, y_tr, y_te, label):
    """Train both models on one train/test split and run the cost-threshold analysis."""
    scaler = StandardScaler().fit(X_tr[["Time", "Amount"]])
    X_tr_lr, X_te_lr = X_tr.copy(), X_te.copy()
    X_tr_lr[["Time", "Amount"]] = scaler.transform(X_tr[["Time", "Amount"]])
    X_te_lr[["Time", "Amount"]] = scaler.transform(X_te[["Time", "Amount"]])

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr_lr, y_tr)
    p_lr = lr.predict_proba(X_te_lr)[:, 1]

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="aucpr", random_state=42, n_jobs=-1,
    ).fit(X_tr, y_tr)
    p_xgb = xgb.predict_proba(X_te)[:, 1]

    pr_auc_lr, pr_auc_xgb = average_precision_score(y_te, p_lr), average_precision_score(y_te, p_xgb)
    roc_auc_lr, roc_auc_xgb = roc_auc_score(y_te, p_lr), roc_auc_score(y_te, p_xgb)

    print(f"\n[{label}]")
    print(f"  Train {len(X_tr):,} / Test {len(X_te):,}  |  test fraud rate {y_te.mean()*100:.3f}%")
    print(f"  {'Logistic (baseline)':22s}  PR-AUC {pr_auc_lr:.3f}   ROC-AUC {roc_auc_lr:.3f}")
    print(f"  {'XGBoost':22s}  PR-AUC {pr_auc_xgb:.3f}   ROC-AUC {roc_auc_xgb:.3f}")

    thresholds = np.linspace(0.01, 0.99, 197)
    costs = [total_cost(y_te, p_xgb, t)[0] for t in thresholds]
    best_thr = thresholds[int(np.argmin(costs))]
    best_cost, fp_b, fn_b = total_cost(y_te, p_xgb, best_thr)
    naive_cost, fp_n, fn_n = total_cost(y_te, p_xgb, 0.5)
    print(f"  Default 0.5 threshold: cost £{naive_cost:,} ({fp_n} FPs, {fn_n} missed frauds)")
    print(f"  Cost-optimal threshold {best_thr:.2f}: cost £{best_cost:,} ({fp_b} FPs, {fn_b} missed frauds)")
    print(f"  Saving vs default: £{naive_cost - best_cost:,} on the test window")

    prec, rec, _ = precision_recall_curve(y_te, p_xgb)
    prec_l, rec_l, _ = precision_recall_curve(y_te, p_lr)

    return dict(
        label=label, xgb=xgb, X_te=X_te, p_xgb=p_xgb, p_lr=p_lr,
        pr_auc_lr=pr_auc_lr, pr_auc_xgb=pr_auc_xgb,
        roc_auc_lr=roc_auc_lr, roc_auc_xgb=roc_auc_xgb,
        thresholds=thresholds, costs=costs, best_thr=best_thr,
        best_cost=best_cost, naive_cost=naive_cost,
        prec=prec, rec=rec, prec_l=prec_l, rec_l=rec_l,
    )


# ---------- random split (stratified, the classic protocol) ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
random_res = evaluate_split(X_tr, X_te, y_tr, y_te, "Random split")

# ---------- time-based split (train on the first 70% chronologically, test on the last 30%) ----------
df_sorted = df.sort_values("Time").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.7)
X_sorted, y_sorted = df_sorted.drop(columns=["Class"]), df_sorted["Class"]
time_res = evaluate_split(
    X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:],
    y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:],
    "Time-based split",
)

# ---------- side-by-side comparison ----------
print("\n" + "=" * 60)
print(f"{'Metric':<28s}{'Random split':>15s}{'Time split':>15s}")
print("-" * 60)
comparison_rows = [
    ("PR-AUC (XGBoost)", f"{random_res['pr_auc_xgb']:.3f}", f"{time_res['pr_auc_xgb']:.3f}"),
    ("PR-AUC (Logistic)", f"{random_res['pr_auc_lr']:.3f}", f"{time_res['pr_auc_lr']:.3f}"),
    ("ROC-AUC (XGBoost)", f"{random_res['roc_auc_xgb']:.3f}", f"{time_res['roc_auc_xgb']:.3f}"),
    ("Cost-optimal threshold", f"{random_res['best_thr']:.2f}", f"{time_res['best_thr']:.2f}"),
    ("Optimal cost", f"£{random_res['best_cost']:,}", f"£{time_res['best_cost']:,}"),
    ("Naive (0.5) cost", f"£{random_res['naive_cost']:,}", f"£{time_res['naive_cost']:,}"),
]
for name, a, b in comparison_rows:
    print(f"{name:<28s}{a:>15s}{b:>15s}")
print("=" * 60)

# ---------- comparison chart ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

axes[0].plot(random_res["rec"], random_res["prec"], label=f"Random (PR-AUC {random_res['pr_auc_xgb']:.3f})")
axes[0].plot(time_res["rec"], time_res["prec"], label=f"Time (PR-AUC {time_res['pr_auc_xgb']:.3f})")
axes[0].set(xlabel="Recall", ylabel="Precision", title="XGBoost PR curve: random vs time split")
axes[0].legend()

axes[1].plot(random_res["thresholds"], random_res["costs"], label="Random split")
axes[1].plot(time_res["thresholds"], time_res["costs"], label="Time split")
axes[1].axvline(random_res["best_thr"], ls="--", color="C0")
axes[1].axvline(time_res["best_thr"], ls="--", color="C1")
axes[1].set(xlabel="Decision threshold", ylabel="Total cost (£)", title="Cost vs threshold: random vs time split")
axes[1].legend()

labels = ["Naive (0.5)", "Cost-optimal"]
x = np.arange(len(labels))
width = 0.35
axes[2].bar(x - width / 2, [random_res["naive_cost"], random_res["best_cost"]], width, label="Random split")
axes[2].bar(x + width / 2, [time_res["naive_cost"], time_res["best_cost"]], width, label="Time split")
axes[2].set_xticks(x, labels)
axes[2].set(ylabel="Total cost (£)", title="Total cost: naive vs optimal threshold")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{OUT}/04_split_comparison.png", dpi=130)
plt.close()

# ---------- primary pipeline charts (random split, unchanged from before) ----------
primary = random_res
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(primary["rec"], primary["prec"], label=f"XGBoost (PR-AUC {primary['pr_auc_xgb']:.3f})")
axes[0].plot(primary["rec_l"], primary["prec_l"], "--", label=f"Logistic (PR-AUC {primary['pr_auc_lr']:.3f})")
axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall (imbalance-aware view)")
axes[0].legend()
axes[1].plot(primary["thresholds"], primary["costs"], color="#c0392b")
axes[1].axvline(primary["best_thr"], ls="--", color="black", label=f"Optimal {primary['best_thr']:.2f}")
axes[1].axvline(0.5, ls=":", color="grey", label="Default 0.5")
axes[1].set(xlabel="Decision threshold", ylabel="Total cost (£)",
            title=f"Business cost vs threshold (FP £{COST_FP}, FN £{COST_FN})")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_pr_curve_and_cost.png", dpi=130)
plt.close()

# ---------- SHAP explainability (on the random-split model) ----------
explainer = shap.TreeExplainer(primary["xgb"])
sample = primary["X_te"].sample(2000, random_state=42)
sv = explainer(sample)

plt.figure()
shap.summary_plot(sv, sample, show=False, max_display=12)
plt.title("Global fraud drivers (SHAP)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_shap_summary.png", dpi=130)
plt.close()

# explain the highest-risk flagged transaction
idx = primary["p_xgb"].argmax()
one = primary["X_te"].iloc[[idx]]
sv_one = explainer(one)
plt.figure()
shap.plots.waterfall(sv_one[0], show=False, max_display=10)
plt.title(f"Why this transaction was flagged (p = {primary['p_xgb'][idx]:.3f})")
plt.tight_layout()
plt.savefig(f"{OUT}/03_shap_single_case.png", dpi=130)
plt.close()

print(f"\nCharts saved to {OUT}/  — done.")
