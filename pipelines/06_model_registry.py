# pipelines/06_model_registry.py
# Registers qualified models in MLflow Model Registry with stage promotion
# Stages: None → Staging → Production

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime
from config.utils import load_config, get_logger, audit_log

logger = get_logger("model_registry")


def run():
    cfg = load_config()
    proc_path  = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["models"])

    logger.info("=" * 60)
    logger.info("STAGE 6: MODEL REGISTRY")
    logger.info("=" * 60)

    # Load training artifacts
    meta_file = model_path / "model_metadata.json"
    qual_file = proc_path / "qualification_report.json"

    with open(meta_file) as f:
        meta = json.load(f)
    with open(qual_file) as f:
        qual = json.load(f)

    if not qual["qualified_for_production"]:
        raise RuntimeError("Cannot register unqualified model.")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    model_name = cfg["model"]["name"]
    run_id     = meta["best_run_id"]

    logger.info(f"Registering model '{model_name}' from run {run_id}...")

    # Register the model (creates new version)
    try:
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = mv.version
        logger.info(f"Registered model version: {version}")
    except Exception as e:
        logger.error(f"MLflow registration failed: {e}")
        logger.info("Falling back to local registry...")
        version = "local-1"

    # Save local registry record (always, as fallback)
    registry_dir = Path(cfg["paths"]["registry"])
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_entry = {
        "model_name": model_name,
        "version": str(version),
        "stage": "Production",
        "registered_at": datetime.utcnow().isoformat() + "Z",
        "mlflow_run_id": run_id,
        "model_type": meta["model_name"],
        "best_f1": meta["best_f1"],
        "iq_result": qual["iq_result"],
        "oq_result": qual["oq_result"],
        "pq_result": qual["pq_result"],
        "feature_columns": meta["feature_columns"],
        "model_path": str(model_path / "best_model.joblib"),
        "qualified_for_production": True,
    }

    registry_file = registry_dir / "registry.json"
    registry_history = []
    if registry_file.exists():
        with open(registry_file) as f:
            try:
                registry_history = json.load(f)
            except Exception:
                registry_history = []

    registry_history.append(registry_entry)
    with open(registry_file, "w") as f:
        json.dump(registry_history, f, indent=2)

    # Write "current production" pointer
    current_file = registry_dir / "current_production.json"
    with open(current_file, "w") as f:
        json.dump(registry_entry, f, indent=2)

    logger.info(f"Local registry updated → {registry_file}")
    logger.info(f"Production pointer → {current_file}")
    logger.info(f"Model '{model_name}' v{version} is now PRODUCTION-ready")

    audit_log("model_registered", {
        "model_name": model_name,
        "version": str(version),
        "stage": "Production",
        "mlflow_run_id": run_id,
        "f1_score": meta["best_f1"],
    })

    logger.info("Stage 6 complete.\n")
    return registry_entry

if __name__ == "__main__":
    run()