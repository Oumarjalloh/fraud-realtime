import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/features.parquet")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
FEAT_LIST_PATH = os.getenv("FEAT_LIST_PATH", "artifacts/feature_list.json")

def main():
    df = pd.read_parquet(FEATURES_PATH)

    # Basic checks
    if "label" not in df.columns:
        raise RuntimeError("features.parquet has no label column. Rebuild features with label.")
    if df["label"].nunique() < 2:
        raise RuntimeError("Label has a single class. Did you load real labels?")

    # Features: drop id/ts/card_id/label
    drop_cols = ["id", "ts", "card_id", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["label"].astype(int)

    # Split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    # Fast LightGBM config
    model = LGBMClassifier(
        n_estimators=2000,          # high but early stopping will cut
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)],
    )

    # Predict
    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    ap = average_precision_score(y_test, p)

    # Default threshold 0.5 just to print a quick report
    yhat = (p >= 0.5).astype(int)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC : {ap:.4f}")
    print(classification_report(y_test, yhat, digits=4))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(FEAT_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("Saved:", MODEL_PATH)
    print("Saved:", FEAT_LIST_PATH)

if __name__ == "__main__":
    main()
