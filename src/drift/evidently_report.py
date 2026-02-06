import os
import json
import pandas as pd
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset

BASELINE_PATH = os.getenv("BASELINE_IN", "artifacts/features.parquet")
OUT_HTML = os.getenv("EVIDENTLY_OUT", "monitoring/evidently_report.html")

POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")

# IMPORTANT: on prend transactions_raw (pas transactions_scored) comme "current"
CURRENT_TABLE = os.getenv("CURRENT_TABLE", "transactions_raw")
CURRENT_LIMIT = int(os.getenv("CURRENT_LIMIT", "20000"))

FEATLIST_PATH = os.getenv("FEATLIST_IN", "artifacts/feature_list.json")
THRESHOLD_PATH = os.getenv("THRESHOLD_IN", "artifacts/threshold.json")


def _load_threshold(default=0.5) -> float:
    try:
        with open(THRESHOLD_PATH, "r") as f:
            return float(json.load(f).get("best_threshold", default))
    except Exception:
        return default


def _dedup(cols):
    # garde l'ordre et enlève les doublons
    return list(dict.fromkeys(cols))


def _add_rollings(df: pd.DataFrame) -> pd.DataFrame:
    # Rebuild rollings like offline features
    def add_rollings(g: pd.DataFrame) -> pd.DataFrame:
        card = g["card_id"].iloc[0]
        g = g.sort_values("ts").copy()
        g = g.set_index("ts")

        g["tx_count_1h"] = g["amount"].rolling("1h").count().astype(float)
        g["tx_count_24h"] = g["amount"].rolling("24h").count().astype(float)

        g["amount_sum_1h"] = g["amount"].rolling("1h").sum().astype(float)
        g["amount_sum_24h"] = g["amount"].rolling("24h").sum().astype(float)

        g["amount_avg_7D"] = g["amount"].rolling("7D").mean().astype(float)

        g = g.reset_index()
        g["card_id"] = card
        return g

    df = df.sort_values(["card_id", "ts"]).copy()

    # time_since_prev_tx
    df["time_since_prev_tx"] = (
        df.groupby("card_id")["ts"].diff().dt.total_seconds().fillna(0.0).astype(float)
    )

    df = df.groupby("card_id", group_keys=False).apply(add_rollings)

    df["amount_avg_7D"] = df["amount_avg_7D"].fillna(df["amount"].median())
    df["amount_vs_avg_7d"] = df["amount"] / (df["amount_avg_7D"] + 1e-6)

    for c in ["tx_count_1h", "tx_count_24h", "amount_sum_1h", "amount_sum_24h", "amount_vs_avg_7d"]:
        df[c] = df[c].fillna(0.0).astype(float)

    return df


def main():
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

    # Baseline
    baseline = pd.read_parquet(BASELINE_PATH).sort_values("ts").reset_index(drop=True)

    # Liste de features attendues
    with open(FEATLIST_PATH, "r") as f:
        feature_cols = json.load(f)

    # Connexion DB
    url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(url)

    # Current depuis transactions_raw (v1..v28 existent)
    # Label peut ne pas exister -> on met NULL
    query = f"""
    SELECT id, ts, card_id, amount,
           v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
           v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
           v21,v22,v23,v24,v25,v26,v27,v28,
           NULL::int as label
    FROM {CURRENT_TABLE}
    ORDER BY ts DESC
    LIMIT {CURRENT_LIMIT}
    """
    current = pd.read_sql(query, engine, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    # rebuild features on current
    current = _add_rollings(current)

    # Harmonise colonnes à comparer : meta + features
    base_cols = ["ts", "card_id", "amount", "label"]
    cols_wanted = _dedup(base_cols + feature_cols)

    # Sécurité: on prend intersection baseline/current
    cols_common = [c for c in cols_wanted if c in baseline.columns and c in current.columns]

    # IMPORTANT: éviter colonnes dupliquées => Evidently bug dtype
    cols_common = _dedup(cols_common)

    ref_evi = baseline[cols_common].copy()
    cur_evi = current[cols_common].copy()

    # Types propres
    if "label" in ref_evi.columns:
        ref_evi["label"] = pd.to_numeric(ref_evi["label"], errors="coerce").fillna(0).astype(int)
    if "label" in cur_evi.columns:
        cur_evi["label"] = pd.to_numeric(cur_evi["label"], errors="coerce").fillna(0).astype(int)

    thr = _load_threshold(0.5)
    print("Evidently columns:", len(cols_common))
    print("Baseline rows:", len(ref_evi), "| Current rows:", len(cur_evi))
    print("Threshold loaded:", thr)

    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    report.run(reference_data=ref_evi, current_data=cur_evi)
    report.save_html(OUT_HTML)
    print("Saved:", OUT_HTML)


if __name__ == "__main__":
    main()
