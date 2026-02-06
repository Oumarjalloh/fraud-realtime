import os
import pandas as pd
from sqlalchemy import create_engine

POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

OUT_PATH = os.getenv("FEATURES_OUT", "artifacts/features.parquet")

def main():
    url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(url)

    query = """
    SELECT id, ts, card_id, amount,
           v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
           v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
           v21,v22,v23,v24,v25,v26,v27,v28,
           label
    FROM transactions_raw
    ORDER BY ts ASC
    """
    df = pd.read_sql(query, engine, parse_dates=["ts"])
    if df.empty:
        raise RuntimeError("transactions_raw is empty. Run ingestion first.")

    df["label"] = df["label"].fillna(0).astype(int)

    df = df.sort_values(["card_id", "ts"]).reset_index(drop=True)

    # time_since_prev_tx (seconds)
    prev_ts = df.groupby("card_id")["ts"].shift(1)
    df["time_since_prev_tx"] = (df["ts"] - prev_ts).dt.total_seconds()
    df["time_since_prev_tx"] = df["time_since_prev_tx"].fillna(999999).clip(0, 999999)

    def add_rollings(g: pd.DataFrame) -> pd.DataFrame:
        card = g.name
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

    df = df.groupby("card_id", group_keys=False).apply(add_rollings)

    df["amount_avg_7D"] = df["amount_avg_7D"].fillna(df["amount"].median())
    df["amount_vs_avg_7d"] = df["amount"] / (df["amount_avg_7D"] + 1e-6)

    for c in ["tx_count_1h","tx_count_24h","amount_sum_1h","amount_sum_24h","amount_vs_avg_7d"]:
        df[c] = df[c].fillna(0.0)

    feature_cols = (
        ["amount"] +
        [f"v{i}" for i in range(1,29)] +
        ["time_since_prev_tx", "tx_count_1h", "tx_count_24h", "amount_sum_1h", "amount_sum_24h", "amount_vs_avg_7d"]
    )

    print("COLUMNS:", list(df.columns))
    out = df[["id","ts","card_id","label"] + feature_cols].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("Saved features to:", OUT_PATH)
    print("Rows:", len(out))

if __name__ == "__main__":
    main()
