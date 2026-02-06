import os
import time
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Fraud Scoring API", version="0.1")

REQUESTS = Counter("requests_total", "Total API requests", ["endpoint"])
FRAUD_PRED = Counter("fraud_predicted_total", "Total fraud predictions")
LATENCY = Histogram("request_latency_ms", "Request latency (ms)")

@app.get("/health")
def health():
    REQUESTS.labels(endpoint="/health").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    REQUESTS.labels(endpoint="/").inc()
    return {"message": "Fraud API running"}
