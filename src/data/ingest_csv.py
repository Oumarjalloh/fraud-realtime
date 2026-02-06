import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

CSV_PATH = os.getenv("CSV_PATH", "data_creditcard.csv")

POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def main():
    url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(url)

    df = pd.read_csv(CSV_PATH)

    # Kaggle names: Time, V1..V28, Amount, Class
    # Normalize columns to lowercase v1..v28 and label
    if "Class" in df.columns:
        df["label"] = df["Class"].astype(int)
    elif "class" in df.columns:
        df["label"] = df["class"].astype(int)
    else:
        raise RuntimeError("No Class column found in CSV (expected 'Class').")

    # Make a timestamp from Time (seconds since start)
    if "Time" in df.columns:
        base = pd.Timestamp("2024-01-01 00:00:00")
        df["ts"] = base + pd.to_timedelta(df["Time"], unit="s")
    elif "ts" not in df.columns:
        df["ts"] = pd.Timestamp("2024-01-01 00:00:00")

    # Build synthetic card_id so we can compute velocity features.
    # This groups transactions into pseudo-cards.
    rng = np.random.default_rng(42)
    n_cards = int(os.getenv("N_CARDS", "5000"))
    df["card_id"] = rng.integers(1, n_cards + 1, size=len(df)).astype(str)

    # Rename V1..V28 to v1..v28
    for i in range(1, 29):
        if f"V{i}" in df.columns:
            df.rename(columns={f"V{i}": f"v{i}"}, inplace=True)
        elif f"v{i}" not in df.columns:
            df[f"v{i}"] = 0.0

    # Amount
    if "Amount" in df.columns:
        df.rename(columns={"Amount": "amount"}, inplace=True)
    elif "amount" not in df.columns:
        raise RuntimeError("No Amount column found in CSV.")

    # Keep only the columns we store
    cols = ["ts", "card_id", "amount"] + [f"v{i}" for i in range(1, 29)] + ["label"]
    out = df[cols].copy()

    # Clear and reload (fast + simple for a project)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE transactions_raw RESTART IDENTITY;"))

    out.to_sql("transactions_raw", engine, if_exists="append", index=False, chunksize=20000)

    print("Loaded rows:", len(out))
    print("Fraud rate:", float(out["label"].mean()))

if __name__ == "__main__":
    main()
