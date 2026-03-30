# pipelines/retrain_trigger.py
# Watches the monitoring API for drift alerts and automatically retrains.
# Run: python pipelines/retrain_trigger.py
#
# In production this would be a cron job or Airflow DAG.
# Locally it polls the monitoring endpoint every N seconds.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import importlib.util
import requests
from pathlib import Path
from datetime import datetime
from config.utils import load_config, get_logger, audit_log

logger = get_logger("retrain_trigger")
cfg    = load_config()

MONITOR_URL      = f"http://localhost:{cfg['monitoring']['port']}"
POLL_INTERVAL_S  = 60          # Check monitoring every 60 seconds
DRIFT_THRESHOLD  = cfg["validation"]["max_drift_score"]   # 0.15 = 15% features drifted
RETRAIN_COOLDOWN = 300         # Don't retrain more than once every 5 minutes

_last_retrain_time = 0


def check_drift_via_api() -> dict | None:
    """Poll monitoring API for latest drift report."""
    try:
        # First trigger a fresh check
        r = requests.post(f"{MONITOR_URL}/drift/run", timeout=30)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.ConnectionError:
        logger.warning("Monitoring API not reachable. Falling back to direct drift check.")
    return None


def check_drift_direct() -> dict:
    """Bypass API and run drift check directly (used when monitor service not running)."""
    from monitoring.monitor import load_dataframes, run_evidently_drift, check_and_alert
    df_ref, df_prod = load_dataframes()
    if df_ref is None:
        return {"dataset_drift_detected": False, "share_drifted_features": 0}
    report = run_evidently_drift(df_ref, df_prod)
    check_and_alert(report)
    return report


def retrain():
    """Re-run the full pipeline to produce a new model."""
    global _last_retrain_time

    now = time.time()
    if now - _last_retrain_time < RETRAIN_COOLDOWN:
        remaining = int(RETRAIN_COOLDOWN - (now - _last_retrain_time))
        logger.info(f"Retrain cooldown active — {remaining}s remaining. Skipping.")
        return False

    logger.info("=" * 60)
    logger.info("RETRAINING TRIGGERED — running full pipeline")
    logger.info("=" * 60)

    audit_log("retrain_triggered", {
        "triggered_at": datetime.utcnow().isoformat() + "Z",
        "reason": "drift_detected",
    })

    ROOT = Path(__file__).parent.parent
    stages = [
        ("01_data_ingestion",      ROOT / "pipelines/01_data_ingestion.py"),
        ("02_data_validation",     ROOT / "pipelines/02_data_validation.py"),
        ("03_feature_engineering", ROOT / "pipelines/03_feature_engineering.py"),
        ("04_model_training",      ROOT / "pipelines/04_model_training.py"),
        ("05_model_validation",    ROOT / "pipelines/05_model_validation.py"),
        ("06_model_registry",      ROOT / "pipelines/06_model_registry.py"),
    ]

    t_start = time.perf_counter()
    completed = []

    for name, path in stages:
        try:
            logger.info(f"  Running {name}...")
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.run()
            completed.append(name)
            logger.info(f"  ✓ {name} done")
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")
            audit_log("retrain_failed", {
                "failed_stage": name,
                "error": str(e),
                "completed_stages": completed,
            })
            return False

    elapsed = time.perf_counter() - t_start
    _last_retrain_time = time.time()

    logger.info(f"\nRetraining complete in {elapsed:.0f}s")
    logger.info("Triggering model hot-reload in serving API...")

    # Tell the serving API to reload the new model
    try:
        r = requests.post(f"http://localhost:{cfg['serving']['port']}/model/reload", timeout=10)
        if r.status_code == 200:
            logger.info("Serving API reloaded new model ✓")
        else:
            logger.warning(f"Serving API reload returned {r.status_code}")
    except requests.exceptions.ConnectionError:
        logger.warning("Serving API not reachable for hot-reload (model saved to disk, restart manually)")

    audit_log("retrain_completed", {
        "elapsed_s":       round(elapsed, 2),
        "completed_stages": completed,
    })
    return True


def run_once():
    """Run a single drift check + retrain if needed. Good for cron/CI."""
    logger.info("Running single drift check...")
    report = check_drift_direct()
    drift_share = report.get("share_drifted_features", 0)
    detected    = report.get("dataset_drift_detected", False)

    logger.info(f"Drift detected: {detected} | Share: {drift_share:.1%} | Threshold: {DRIFT_THRESHOLD:.0%}")

    if detected and drift_share > DRIFT_THRESHOLD:
        logger.warning(f"Drift exceeds threshold ({drift_share:.1%} > {DRIFT_THRESHOLD:.0%}) — retraining now")
        success = retrain()
        return success
    else:
        logger.info("Drift within acceptable limits — no retraining needed")
        return False


def run_continuous():
    """Continuously poll for drift and retrain when needed."""
    logger.info("=" * 60)
    logger.info("RETRAIN TRIGGER — Continuous Mode")
    logger.info(f"  Poll interval : {POLL_INTERVAL_S}s")
    logger.info(f"  Drift threshold: {DRIFT_THRESHOLD:.0%} of features")
    logger.info(f"  Retrain cooldown: {RETRAIN_COOLDOWN}s")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    check_count   = 0
    retrain_count = 0

    while True:
        try:
            check_count += 1
            logger.info(f"\n[Check #{check_count}] {datetime.utcnow().isoformat()}Z")

            # Try API first, fall back to direct
            report = check_drift_via_api() or check_drift_direct()
            drift_share = report.get("share_drifted_features", 0)
            detected    = report.get("dataset_drift_detected", False)
            drifted     = report.get("drifted_columns", [])

            logger.info(f"  Drift: {detected} | Share: {drift_share:.1%} | Cols: {drifted}")

            if detected and drift_share > DRIFT_THRESHOLD:
                logger.warning(f"⚠ Drift threshold exceeded — initiating retrain")
                if retrain():
                    retrain_count += 1
                    logger.info(f"✓ Retrain #{retrain_count} completed")
            else:
                logger.info("✓ No action needed")

            time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            logger.info(f"\nStopped after {check_count} checks, {retrain_count} retrains.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in trigger loop: {e}")
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pharma MLOps Retrain Trigger")
    parser.add_argument(
        "--mode", choices=["once", "continuous"], default="once",
        help="'once' = single check + retrain if needed | 'continuous' = poll forever"
    )
    args = parser.parse_args()

    if args.mode == "once":
        run_once()
    else:
        run_continuous()
