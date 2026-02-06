# Fraud Realtime — Détection de fraude temps réel (API + Monitoring + Drift)

Projet end-to-end de détection de fraude en temps réel :
- Ingestion de transactions dans PostgreSQL (`transactions_raw`)
- Génération de features offline (rolling windows 1h/24h/7D)
- Entraînement d’un modèle LightGBM (binaire fraude / non fraude)
- Choix automatique d’un **seuil** (threshold) basé sur un **coût FP/FN**
- API FastAPI exposant `/score`, `/health`, `/metrics`
- Monitoring Prometheus + dashboards Grafana
- Rapport de drift (Evidently) exporté en HTML

---

## 1) Stack technique

- **Python 3.12**
- **PostgreSQL 16**
- **FastAPI + Uvicorn**
- **LightGBM** (modèle)
- **Prometheus** (scrape metrics)
- **Grafana** (dashboards)
- **Evidently** (drift report)
- **Docker Compose** (orchestration)

---

## 2) Structure du repo

