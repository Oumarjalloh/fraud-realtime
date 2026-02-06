import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FEATURES_PATH = os.getenv("FEATURES_IN", "artifacts/features.parquet")
MODEL_PATH = os.getenv("MODEL_IN", "artifacts/model.pkl")
FEATLIST_PATH = os.getenv("FEATLIST_IN", "artifacts/feature_list.json")
OUT_PATH = os.getenv("THRESHOLD_OUT", "artifacts/threshold.json")

# Business costs (tune these)
COST_FP = float(os.getenv("COST_FP", "5"))     # false positive: friction / manual review
COST_FN = float(os.getenv("COST_FN", "200"))   # false negative: missed fraud

def compute_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    cost = fp * COST_FP + fn * COST_FN
    return cost, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def main():
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values("ts").reset_index(drop=True)

    if "label" not in df.columns:
        raise RuntimeError("No label in features.parquet. Rebuild features with label.")
    if df["label"].nunique() < 2:
        raise RuntimeError("Label has a single class. Did you load real labels?")

    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()

    y = test_df["label"].astype(int)

    with open(FEATLIST_PATH, "r") as f:
        feature_cols = json.load(f)

    X = test_df[feature_cols]
    model = joblib.load(MODEL_PATH)
    p = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    best = None

    for t in thresholds:
        yhat = (p >= t).astype(int)
        cost, cm = compute_cost(y, yhat)
        item = {"threshold": float(t), "cost": float(cost), "confusion": cm}
        if best is None or item["cost"] < best["cost"]:
            best = item

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "best_threshold": best["threshold"],
                "best_cost": best["cost"],
                "confusion_at_best": best["confusion"],
                "cost_fp": COST_FP,
                "cost_fn": COST_FN,
            },
            f,
            indent=2,
        )

    print("Saved:", OUT_PATH)
    print("Best threshold:", best["threshold"])
    print("Best cost:", best["cost"])
    print("Confusion:", best["confusion"])

if __name__ == "__main__":
    main()
