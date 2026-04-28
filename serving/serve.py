import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
import uvicorn
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from config.utils import load_config, get_logger, audit_log

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

def _metric(cls, name, desc, labels=None, **kwargs):
    try:
        return cls(name, desc, labels or [], **kwargs) if labels else cls(name, desc, **kwargs)
    except ValueError:
        collectors = list(REGISTRY._names_to_collectors.values())
        for c in collectors:
            if hasattr(c, '_name') and c._name in (name, name + '_total'):
                return c
        return cls(name, desc, labels or [], **kwargs) if labels else cls(name, desc, **kwargs)

logger  = get_logger("serving")
cfg     = load_config()
app     = FastAPI(
    title="Pharma MLOps - Drug Efficacy Prediction API",
    version="1.0.0"
)

MODEL        = None
FEATURE_COLS = None
REGISTRY     = None
PREDICTION_LOG = []

PREDICTIONS_TOTAL = _metric(Counter, "pharma_predictions_total", "Total predictions made", ["prediction_class", "confidence"])
PREDICTION_LATENCY = _metric(Histogram, "pharma_prediction_latency_seconds", "Prediction request latency", buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
MODEL_INFO = _metric(Gauge, "pharma_model_info", "Current model metadata", ["model_name", "version"])
REQUESTS_TOTAL = _metric(Counter, "pharma_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])


class DrugTrialInput(BaseModel):
    mol_weight:             float = Field(..., ge=50,  le=1000)
    logp:                   float = Field(..., ge=-5,  le=10)
    tpsa:                   float = Field(..., ge=0,   le=300)
    hbd:                    float = Field(..., ge=0,   le=20)
    hba:                    float = Field(..., ge=0,   le=20)
    rotatable_bonds:        float = Field(default=3,   ge=0, le=30)
    aromatic_rings:         float = Field(default=2,   ge=0, le=10)
    patient_age:            float = Field(..., ge=18,  le=100)
    patient_weight_kg:      float = Field(..., ge=30,  le=200)
    creatinine_clearance:   float = Field(default=90,  ge=10, le=200)
    is_male:                float = Field(default=1,   ge=0, le=1)
    baseline_crp:           float = Field(default=5,   ge=0, le=200)
    baseline_il6:           float = Field(default=10,  ge=0, le=500)
    baseline_tnfa:          float = Field(default=15,  ge=0, le=500)
    baseline_wbc:           float = Field(default=7,   ge=0, le=50)
    dose_mg:                float = Field(...)
    treatment_days:         float = Field(..., ge=1,   le=365)
    sample_id:              Optional[str] = Field(default=None)


class BatchInput(BaseModel):
    samples: List[DrugTrialInput]


class PredictionResponse(BaseModel):
    sample_id:               Optional[str]
    prediction:              int
    probability_effective:   float
    probability_ineffective: float
    confidence:              str
    predicted_at:            str


def load_model():
    global MODEL, FEATURE_COLS, REGISTRY
    registry_file = Path(cfg["paths"]["registry"]) / "current_production.json"
    if not registry_file.exists():
        raise RuntimeError("No production model found. Run the full pipeline first.")
    with open(registry_file) as f:
        REGISTRY = json.load(f)
    MODEL = joblib.load(REGISTRY["model_path"])
    FEATURE_COLS = REGISTRY["feature_columns"]
    MODEL_INFO.labels(
        model_name=REGISTRY["model_name"],
        version=str(REGISTRY["version"])
    ).set(1)
    logger.info(f"Model loaded: {REGISTRY['model_name']} v{REGISTRY['version']}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["ro5_mw_ok"]   = (feat["mol_weight"] <= 500).astype(int)
    feat["ro5_logp_ok"] = (feat["logp"] <= 5).astype(int)
    feat["ro5_hbd_ok"]  = (feat["hbd"] <= 5).astype(int)
    feat["ro5_hba_ok"]  = (feat["hba"] <= 10).astype(int)
    feat["ro5_violations"] = 4 - (feat["ro5_mw_ok"] + feat["ro5_logp_ok"] +
                                   feat["ro5_hbd_ok"] + feat["ro5_hba_ok"])
    feat["is_drug_like"] = (feat["ro5_violations"] <= 1).astype(int)
    feat["bioavailability_score"] = (
        (1 - feat["ro5_violations"] / 4) * 0.5 +
        ((feat["tpsa"] < 140).astype(float)) * 0.3 +
        ((feat["rotatable_bonds"] < 10).astype(float)) * 0.2
    ).clip(0, 1)
    feat["inflammation_score"] = (
        (feat["baseline_crp"]  / 80)  * 0.4 +
        (feat["baseline_il6"]  / 200) * 0.35 +
        (feat["baseline_tnfa"] / 300) * 0.25
    ).clip(0, 1)
    feat["renal_adjusted_dose"] = (feat["dose_mg"] * (feat["creatinine_clearance"] / 90)).clip(0, 800)
    feat["bsa_proxy"]           = (feat["patient_weight_kg"] / 70) ** 0.5
    feat["dose_per_bsa"]        = feat["dose_mg"] / feat["bsa_proxy"]
    feat["treatment_intensity"] = feat["dose_mg"] * feat["treatment_days"] / 1000
    feat["age_metabolism_factor"] = np.where(
        feat["patient_age"] < 30, 1.2,
        np.where(feat["patient_age"] > 65, 0.7, 1.0)
    )
    for col in ["baseline_crp", "baseline_il6", "baseline_tnfa"]:
        feat[f"log_{col}"] = np.log1p(feat[col])
    return feat


def predict_single(sample: DrugTrialInput) -> dict:
    import time
    start = time.perf_counter()

    row = pd.DataFrame([sample.model_dump()])
    row = add_engineered_features(row)
    available = [c for c in FEATURE_COLS if c in row.columns]
    X = row[available].fillna(0)
    pred  = int(MODEL.predict(X)[0])
    prob  = MODEL.predict_proba(X)[0]
    p_eff = round(float(prob[1]), 4)
    p_inf = round(float(prob[0]), 4)
    conf  = "high" if max(prob) >= 0.80 else ("medium" if max(prob) >= 0.60 else "low")

    latency = time.perf_counter() - start
    PREDICTION_LATENCY.observe(latency)
    PREDICTIONS_TOTAL.labels(
        prediction_class=str(pred),
        confidence=conf
    ).inc()

    return {
        "sample_id":              sample.sample_id or f"INFER_{datetime.utcnow().strftime('%H%M%S%f')}",
        "prediction":             pred,
        "probability_effective":  p_eff,
        "probability_ineffective":p_inf,
        "confidence":             conf,
        "predicted_at":           datetime.utcnow().isoformat() + "Z",
    }


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  MODEL is not None,
        "model_name":    REGISTRY["model_name"] if REGISTRY else None,
        "model_version": REGISTRY["version"]    if REGISTRY else None,
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }


@app.get("/model/info")
def model_info():
    if not REGISTRY:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return REGISTRY


@app.post("/predict", response_model=PredictionResponse)
def predict(sample: DrugTrialInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        result = predict_single(sample)
        PREDICTION_LOG.append(result)
        REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="200").inc()
        audit_log("prediction_made", {
            "sample_id":  result["sample_id"],
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }, actor="serving_api")
        return result
    except Exception as e:
        REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    results = []
    for sample in batch.samples:
        try:
            results.append(predict_single(sample))
        except Exception as e:
            results.append({"error": str(e), "sample_id": sample.sample_id})
    PREDICTION_LOG.extend(results)
    audit_log("batch_prediction_made", {"n_samples": len(batch.samples)}, actor="serving_api")
    return {"predictions": results, "count": len(results)}


@app.get("/predictions/recent")
def recent_predictions(limit: int = 50):
    return {"predictions": PREDICTION_LOG[-limit:], "total": len(PREDICTION_LOG)}


@app.post("/model/reload")
def reload_model():
    try:
        load_model()
        return {"status": "reloaded", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = cfg["serving"]["host"]
    port = cfg["serving"]["port"]
    logger.info(f"Starting serving API on {host}:{port}")
    uvicorn.run("serving.serve:app", host=host, port=port, reload=False)