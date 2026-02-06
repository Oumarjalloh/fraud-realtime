import os
import json
import time
import joblib
import numpy as np
from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import psycopg2
from psycopg2.extras import RealDictCursor

from prometheus_client import Counter

FRAUD_DECISIONS = Counter(
    "fraud_api_decisions_total",
    "Fraud decisions",
    ["is_fraud"]
)
FRAUD_DECISIONS.labels(is_fraud=str(is_fraud).lower()).inc()



APP_NAME = "fraud-realtime-api"

MODEL_PATH = os.getenv("MODEL_PATH", "/app/artifacts/model.pkl")
FEATLIST_PATH = os.getenv("FEATLIST_PATH", "/app/artifacts/feature_list.json")
THRESH_PATH = os.getenv("THRESH_PATH", "/app/artifacts/threshold.json")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fraud_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

# Prometheus metrics
REQ_COUNT = Counter("fraud_api_requests_total", "Total API requests", ["endpoint"])
REQ_LAT = Histogram("fraud_api_latency_seconds", "API latency", ["endpoint"])
FRAUD_RATE = Gauge("fraud_api_last_is_fraud", "Last decision (1 fraud, 0 legit)")

app = FastAPI(title=APP_NAME, version=MODEL_VERSION)

class ScoreRequest(BaseModel):
    # Minimal fields. Keep same names as features.parquet (except id/ts/label)
    card_id: Optional[str] = None
    amount: float = Field(..., ge=0)
    time_since_prev_tx: Optional[float] = 0.0
    tx_count_1h: Optional[float] = 0.0
    tx_count_24h: Optional[float] = 0.0
    amount_sum_1h: Optional[float] = 0.0
    amount_sum_24h: Optional[float] = 0.0
    amount_avg_7D: Optional[float] = 0.0
    amount_vs_avg_7d: Optional[float] = 0.0

    # v1..v28
    v1: float = 0.0; v2: float = 0.0; v3: float = 0.0; v4: float = 0.0; v5: float = 0.0
    v6: float = 0.0; v7: float = 0.0; v8: float = 0.0; v9: float = 0.0; v10: float = 0.0
    v11: float = 0.0; v12: float = 0.0; v13: float = 0.0; v14: float = 0.0; v15: float = 0.0
    v16: float = 0.0; v17: float = 0.0; v18: float = 0.0; v19: float = 0.0; v20: float = 0.0
    v21: float = 0.0; v22: float = 0.0; v23: float = 0.0; v24: float = 0.0; v25: float = 0.0
    v26: float = 0.0; v27: float = 0.0; v28: float = 0.0

def get_pg_conn():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )

# Load artifacts once at startup
with open(FEATLIST_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

with open(THRESH_PATH, "r") as f:
    THRESH = json.load(f)
    BEST_THRESHOLD = float(THRESH["best_threshold"])

MODEL = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "threshold": BEST_THRESHOLD}

@app.post("/score")
def score(req: ScoreRequest) -> Dict[str, Any]:
    t0 = time.time()
    REQ_COUNT.labels(endpoint="/score").inc()

    payload = req.dict()
    # Build vector in correct order
    x = np.array([[payload.get(c, 0.0) for c in FEATURE_COLS]], dtype=float)
    prob = float(MODEL.predict_proba(x)[:, 1][0])
    is_fraud = bool(prob >= BEST_THRESHOLD)

    latency = (time.time() - t0) * 1000.0
    FRAUD_RATE.set(1.0 if is_fraud else 0.0)
    REQ_LAT.labels(endpoint="/score").observe(latency / 1000.0)

    # Persist decision (best effort)
    try:
        conn = get_pg_conn()
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO transactions_scored(card_id, amount, score, threshold, is_fraud, model_version, latency_ms)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        payload.get("card_id"),
                        float(payload.get("amount", 0.0)),
                        prob,
                        BEST_THRESHOLD,
                        is_fraud,
                        MODEL_VERSION,
                        latency,
                    ),
                )
        conn.close()
    except Exception:
        pass

    return {
        "score": prob,
        "threshold": BEST_THRESHOLD,
        "is_fraud": is_fraud,
        "model_version": MODEL_VERSION,
        "latency_ms": latency,
    }

@app.get("/metrics")
def metrics():
    REQ_COUNT.labels(endpoint="/metrics").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
