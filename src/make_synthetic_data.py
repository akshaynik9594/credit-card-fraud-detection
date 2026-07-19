"""
Synthetic fallback generator for the ULB credit-card fraud dataset schema.

The real project should use the actual dataset:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  (creditcard.csv)
Download it and place it at data/creditcard.csv — train.py will use it
automatically. This generator only exists so the pipeline runs end-to-end
without the download (e.g. for CI or quick demos).

Schema matched: Time, V1..V28 (PCA-like components), Amount, Class (1 = fraud).
Fraud rate ~0.17%, mirroring the real dataset's extreme imbalance.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 120_000
FRAUD_RATE = 0.0017
n_fraud = int(N * FRAUD_RATE)

# legitimate transactions: V's ~ standard normal-ish
X_legit = rng.normal(0, 1, size=(N - n_fraud, 28))

# fraud: partially shifted means on a subset of components, with heavy
# overlap so performance resembles the real dataset (PR-AUC ~0.8, not 1.0)
shift = np.zeros(28)
comp = [1, 3, 4, 9, 10, 11, 13, 16]
shift[comp] = rng.choice([-1, 1], len(comp)) * rng.uniform(0.5, 1.1, len(comp))
X_fraud = rng.normal(0, 1.6, size=(n_fraud, 28))
# only ~65% of frauds follow the shifted pattern; the rest look legitimate
mask = rng.random(n_fraud) < 0.65
X_fraud[mask] += shift * rng.uniform(0.5, 1.5, (mask.sum(), 1))

X = np.vstack([X_legit, X_fraud])
y = np.array([0] * (N - n_fraud) + [1] * n_fraud)

amount = np.where(
    y == 1,
    rng.lognormal(4.4, 1.1, N),   # fraud skews to higher amounts
    rng.lognormal(3.2, 1.2, N),
).round(2)
time = np.sort(rng.uniform(0, 172_800, N))  # two days of seconds, like the real set

df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", time)
df["Amount"] = amount
df["Class"] = y
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/creditcard_synthetic.csv", index=False)
print(f"Synthetic set: {len(df):,} rows, {df['Class'].sum():,} frauds ({df['Class'].mean()*100:.3f}%)")
