# monitoring/monitor.py
# Drift detection using Evidently AI (v0.7+) with scipy KS fallback
# Run: python monitoring/monitor.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import threading
import pandas as pd
import numpy as np
import uvicorn
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Evidently 0.7+ API
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False

from config.utils import load_config, get_logger, audit_log

logger = get_logger("monitoring")
cfg    = load_config()
app    = FastAPI(title="Pharma MLOps - Monitoring API")

# In-memory state
DRIFT_HISTORY = []
LAST_REPORT   = None

MONITOR_FEATURES = [
    "mol_weight", "logp", "tpsa", "patient_age", "patient_weight_kg",
    "creatinine_clearance", "baseline_crp", "baseline_il6", "dose_mg",
    "treatment_days", "bioavailability_score", "inflammation_score",
]


def load_dataframes():
    proc_path = Path(cfg["paths"]["processed_data"])
    ref_file  = proc_path / "reference.csv"
    prod_file = proc_path / "features_production.csv"
    if not ref_file.exists() or not prod_file.exists():
        logger.error("Reference or production data not found. Run pipeline first.")
        return None, None
    return pd.read_csv(ref_file), pd.read_csv(prod_file)


def run_evidently_drift(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    """
    Run drift detection.
    Primary:  Evidently 0.7 Report API (DataDriftPreset)
    Fallback: scipy KS test (always runs as baseline)
    """
    cols = [c for c in MONITOR_FEATURES if c in df_ref.columns and c in df_prod.columns]
    ref  = df_ref[cols].copy()
    curr = df_prod[cols].copy()

    # --- Per-column KS drift (reliable baseline, always runs) ---
    col_drift = {}
    for col in cols:
        r = ref[col].dropna().values
        c = curr[col].dropna().values
        if len(r) > 1 and len(c) > 1:
            stat, pval = ks_2samp(r, c)
            col_drift[col] = {
                "ks_statistic": round(float(stat), 4),
                "p_value":      round(float(pval), 4),
                "drifted":      bool(pval < 0.05),
            }
        else:
            col_drift[col] = {"ks_statistic": None, "p_value": None, "drifted": False}

    drifted_cols   = [c for c, d in col_drift.items() if d["drifted"]]
    n_drifted      = len(drifted_cols)
    drift_share    = n_drifted / len(cols) if cols else 0.0
    drift_detected = n_drifted > 0

    report = {
        "evaluated_at":           datetime.utcnow().isoformat() + "Z",
        "dataset_drift_detected": drift_detected,
        "n_drifted_features":     n_drifted,
        "n_total_features":       len(cols),
        "share_drifted_features": round(drift_share, 4),
        "drifted_columns":        drifted_cols,
        "per_column_drift":       col_drift,
        "n_reference_samples":    len(ref),
        "n_current_samples":      len(curr),
        "method":                 "scipy_ks",
    }

    # --- Evidently 0.7 enrichment ---
    if EVIDENTLY_OK:
        try:
            ev_report = Report(metrics=[DataDriftPreset()])
            snap      = ev_report.run(current_data=curr, reference_data=ref)
            metrics   = snap.dict().get("metrics", [])

            for m in metrics:
                name = m.get("metric_name", "")
                val  = m.get("value")

                # DriftedColumnsCount → dataset-level summary
                if "DriftedColumnsCount" in name and isinstance(val, dict):
                    report["evidently_drift_share"]   = round(float(val.get("share", drift_share)), 4)
                    report["evidently_drifted_count"] = int(val.get("count", n_drifted))

                # ValueDrift → per-column p-value from Evidently
                elif "ValueDrift" in name and isinstance(val, (int, float)):
                    try:
                        col_name = name.split("column=")[1].split(",")[0]
                        if col_name in col_drift:
                            col_drift[col_name]["evidently_pval"] = round(float(val), 6)
                    except IndexError:
                        pass

            report["method"] = "evidently_v07+scipy_ks"
            logger.info("Evidently 0.7 enrichment applied successfully.")
        except Exception as e:
            logger.warning(f"Evidently enrichment skipped (KS only): {e}")

    return report


def compute_stats(df: pd.DataFrame, cols: list) -> dict:
    stats = {}
    for col in cols:
        if col in df.columns:
            s = df[col].describe()
            stats[col] = {
                "mean":   round(float(s["mean"]), 3),
                "std":    round(float(s["std"]),  3),
                "min":    round(float(s["min"]),  3),
                "max":    round(float(s["max"]),  3),
                "median": round(float(df[col].median()), 3),
            }
    return stats


def check_and_alert(report: dict):
    threshold = cfg["validation"]["max_drift_score"]
    share     = report.get("share_drifted_features", 0)
    detected  = report.get("dataset_drift_detected", False)
    drifted   = report.get("drifted_columns", [])

    if detected or share > threshold:
        logger.warning("=" * 55)
        logger.warning("⚠  DRIFT ALERT: Significant data drift detected!")
        logger.warning(f"   Drifted features : {report.get('n_drifted_features')} / {report.get('n_total_features')}")
        logger.warning(f"   Drift share      : {share:.2%}  (threshold: {threshold:.0%})")
        logger.warning(f"   Drifted columns  : {drifted}")
        logger.warning("   → Retraining is recommended — run python run.py")
        logger.warning("=" * 55)
        audit_log("drift_alert", {
            "share_drifted":       round(share, 4),
            "n_drifted":           report.get("n_drifted_features"),
            "drifted_columns":     drifted,
            "retrain_recommended": True,
        })
    else:
        logger.info(
            f"✓ No significant drift. "
            f"Drifted: {report.get('n_drifted_features', 0)}/{report.get('n_total_features', '?')} "
            f"features ({share:.1%})"
        )


# ── FastAPI endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "evidently": EVIDENTLY_OK,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/drift/latest")
def get_latest_drift():
    if not DRIFT_HISTORY:
        return {"message": "No drift evaluation run yet. POST /drift/run to start."}
    return DRIFT_HISTORY[-1]


@app.get("/drift/history")
def get_drift_history():
    return {"history": DRIFT_HISTORY, "count": len(DRIFT_HISTORY)}


@app.post("/drift/run")
def run_drift_check():
    global LAST_REPORT
    df_ref, df_prod = load_dataframes()
    if df_ref is None:
        return JSONResponse(status_code=503, content={"error": "Data not available. Run pipeline first."})
    try:
        logger.info("Running drift detection...")
        report = run_evidently_drift(df_ref, df_prod)
        report["ref_stats"]  = compute_stats(df_ref,  MONITOR_FEATURES)
        report["prod_stats"] = compute_stats(df_prod, MONITOR_FEATURES)
        check_and_alert(report)
        DRIFT_HISTORY.append(report)
        LAST_REPORT = report

        monitoring_dir = Path("monitoring")
        monitoring_dir.mkdir(exist_ok=True)
        with open(monitoring_dir / "latest_drift_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/stats/reference")
def reference_stats():
    df_ref, _ = load_dataframes()
    if df_ref is None:
        return JSONResponse(status_code=503, content={"error": "Data not available"})
    return {
        "n_samples":          len(df_ref),
        "stats":              compute_stats(df_ref, MONITOR_FEATURES),
        "label_distribution": df_ref["efficacy_label"].value_counts().to_dict()
                              if "efficacy_label" in df_ref.columns else {},
    }


@app.get("/stats/production")
def production_stats():
    _, df_prod = load_dataframes()
    if df_prod is None:
        return JSONResponse(status_code=503, content={"error": "Data not available"})
    return {
        "n_samples": len(df_prod),
        "stats":     compute_stats(df_prod, MONITOR_FEATURES),
    }


def _background_monitoring_loop():
    """Periodic drift checks in a background thread."""
    import schedule as sched
    interval = cfg["monitoring"]["drift_check_interval_seconds"]

    def job():
        df_ref, df_prod = load_dataframes()
        if df_ref is not None:
            try:
                rep = run_evidently_drift(df_ref, df_prod)
                check_and_alert(rep)
                DRIFT_HISTORY.append(rep)
            except Exception as e:
                logger.error(f"Background drift check error: {e}")

    sched.every(interval).seconds.do(job)
    logger.info(f"Background drift monitor started — checking every {interval}s")
    job()   # Run immediately on startup
    while True:
        sched.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    t = threading.Thread(target=_background_monitoring_loop, daemon=True)
    t.start()

    host = cfg["monitoring"]["host"]
    port = cfg["monitoring"]["port"]
    logger.info(f"Starting monitoring API on {host}:{port}")
    uvicorn.run("monitoring.monitor:app", host=host, port=port, reload=False)
