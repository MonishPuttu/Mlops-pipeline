# serving/serve.py
# FastAPI REST API for model inference
# Run: python serving/serve.py

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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from config.utils import load_config, get_logger, audit_log

logger  = get_logger("serving")
cfg     = load_config()
app     = FastAPI(
    title="Pharma MLOps - Drug Efficacy Prediction API",
    description="Predicts drug efficacy for pharma clinical use cases",
    version="1.0.0"
)

# Global model state
MODEL       = None
FEATURE_COLS = None
REGISTRY    = None
PREDICTION_LOG = []  # In-memory log; production would use a DB


class DrugTrialInput(BaseModel):
    """Input schema for a single drug trial prediction."""
    mol_weight:             float = Field(..., ge=50,  le=1000,  description="Molecular weight (Da)")
    logp:                   float = Field(..., ge=-5,  le=10,    description="Lipophilicity (LogP)")
    tpsa:                   float = Field(..., ge=0,   le=300,   description="Topological polar surface area")
    hbd:                    float = Field(..., ge=0,   le=20,    description="Hydrogen bond donors")
    hba:                    float = Field(..., ge=0,   le=20,    description="Hydrogen bond acceptors")
    rotatable_bonds:        float = Field(default=3,  ge=0,  le=30)
    aromatic_rings:         float = Field(default=2,  ge=0,  le=10)
    patient_age:            float = Field(..., ge=18,  le=100,   description="Patient age (years)")
    patient_weight_kg:      float = Field(..., ge=30,  le=200,   description="Patient weight (kg)")
    creatinine_clearance:   float = Field(default=90, ge=10, le=200, description="Renal function (mL/min)")
    is_male:                float = Field(default=1,  ge=0,  le=1)
    baseline_crp:           float = Field(default=5,  ge=0,  le=200, description="C-reactive protein (mg/L)")
    baseline_il6:           float = Field(default=10, ge=0,  le=500)
    baseline_tnfa:          float = Field(default=15, ge=0,  le=500)
    baseline_wbc:           float = Field(default=7,  ge=0,  le=50)
    dose_mg:                float = Field(..., description="Dose in mg")
    treatment_days:         float = Field(..., ge=1,  le=365, description="Treatment duration (days)")
    sample_id:              Optional[str] = Field(default=None, description="Optional sample identifier")


class BatchInput(BaseModel):
    samples: List[DrugTrialInput]


class PredictionResponse(BaseModel):
    sample_id: Optional[str]
    prediction: int
    probability_effective: float
    probability_ineffective: float
    confidence: str
    predicted_at: str


def load_model():
    global MODEL, FEATURE_COLS, REGISTRY
    registry_file = Path(cfg["paths"]["registry"]) / "current_production.json"
    if not registry_file.exists():
        raise RuntimeError("No production model found. Run the full pipeline first.")

    with open(registry_file) as f:
        REGISTRY = json.load(f)

    model_path = REGISTRY["model_path"]
    MODEL = joblib.load(model_path)
    FEATURE_COLS = REGISTRY["feature_columns"]
    logger.info(f"Model loaded: {REGISTRY['model_name']} v{REGISTRY['version']} (F1={REGISTRY['best_f1']})")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features matching the training pipeline."""
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
    row = pd.DataFrame([sample.model_dump()])
    row = add_engineered_features(row)
    available = [c for c in FEATURE_COLS if c in row.columns]
    X = row[available].fillna(0)
    pred   = int(MODEL.predict(X)[0])
    prob   = MODEL.predict_proba(X)[0]
    p_eff  = round(float(prob[1]), 4)
    p_ineff= round(float(prob[0]), 4)
    conf   = "high" if max(prob) >= 0.80 else ("medium" if max(prob) >= 0.60 else "low")
    return {
        "sample_id":              sample.sample_id or f"INFER_{datetime.utcnow().strftime('%H%M%S%f')}",
        "prediction":             pred,
        "probability_effective":  p_eff,
        "probability_ineffective":p_ineff,
        "confidence":             conf,
        "predicted_at":           datetime.utcnow().isoformat() + "Z",
    }


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_name":   REGISTRY["model_name"] if REGISTRY else None,
        "model_version":REGISTRY["version"]    if REGISTRY else None,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
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
        audit_log("prediction_made", {"sample_id": result["sample_id"], "prediction": result["prediction"],
                                       "confidence": result["confidence"]}, actor="serving_api")
        return result
    except Exception as e:
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
    """Returns recent predictions (for monitoring dashboard)."""
    return {"predictions": PREDICTION_LOG[-limit:], "total": len(PREDICTION_LOG)}


@app.post("/model/reload")
def reload_model():
    """Hot-reload model from registry (used after retraining)."""
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
