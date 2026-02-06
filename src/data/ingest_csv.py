import os
import pandas as pd
from sqlalchemy import create_engine

POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def main():
    csv_path = "./data_creditcard.csv"
    df = pd.read_csv(csv_path)

   
    base = pd.Timestamp("2024-01-01 00:00:00")
    df["ts"] = base + pd.to_timedelta(df["Time"], unit="s")
    df["card_id"] = "card_" + (df.index % 5000).astype(str)

    # Rename label
    df["label"] = df["Class"].astype(int)
    df.rename(columns={"Amount": "amount"}, inplace=True)

    cols = ["ts", "card_id", "amount"] + [f"V{i}" for i in range(1, 29)] + ["label"]
    df = df[cols]

    # Lowercase V columns to match SQL schema v1..v28
    df.columns = [c.lower() if c.startswith("V") else c for c in df.columns]
    df.columns = [c.lower() for c in df.columns]

    url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(url)

    df.to_sql("transactions_raw", engine, if_exists="append", index=False, chunksize=20000)
    print(f"Inserted {len(df)} rows into transactions_raw")

if __name__ == "__main__":
    main()
