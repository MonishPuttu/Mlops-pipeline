#!/usr/bin/env python3
# run.py - Simple all-in-one runner (no import tricks)
# Usage: python run.py

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.utils import load_config, get_logger, audit_log, ensure_dirs

logger = get_logger("runner")

def banner(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def run_stage(label, module_path):
    banner(label)
    t = time.perf_counter()
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.run()
    elapsed = time.perf_counter() - t
    print(f"\n✓ {label} done in {elapsed:.1f}s")
    return result

def main():
    ensure_dirs()
    cfg = load_config()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║           PHARMA MLOps PIPELINE — LOCAL RUNNER                  ║
║  Drug Efficacy Prediction · End-to-End ML Pipeline              ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    started = datetime.utcnow().isoformat()
    audit_log("pipeline_started", {"started_at": started})
    pipeline_t = time.perf_counter()

    stages = [
        ("Stage 1/6 · Data Ingestion",       ROOT / "pipelines/01_data_ingestion.py"),
        ("Stage 2/6 · Data Validation",       ROOT / "pipelines/02_data_validation.py"),
        ("Stage 3/6 · Feature Engineering",   ROOT / "pipelines/03_feature_engineering.py"),
        ("Stage 4/6 · Model Training",        ROOT / "pipelines/04_model_training.py"),
        ("Stage 5/6 · IQ/OQ/PQ Validation",  ROOT / "pipelines/05_model_validation.py"),
        ("Stage 6/6 · Model Registry",        ROOT / "pipelines/06_model_registry.py"),
    ]

    for label, path in stages:
        try:
            run_stage(label, str(path))
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            print("Pipeline halted. Fix the error and re-run.")
            sys.exit(1)

    total = time.perf_counter() - pipeline_t
    audit_log("pipeline_completed", {"total_s": round(total, 2), "all_passed": True})

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   ALL STAGES COMPLETED in {total:.0f}s                             
║                                                                  ║
║  Now start the services (each in a separate terminal):           ║
║                                                                  ║
║  1. MLflow UI (reads local ./mlruns):                            ║
║     mlflow ui --host 127.0.0.1 --port 5000                      ║
║                                                                  ║
║  2. Model Serving API (port 8000):                               ║
║     python serving/serve.py                                      ║
║                                                                  ║
║  3. Monitoring Service (port 8001):                              ║
║     python monitoring/monitor.py                                  ║
║                                                                  ║
║  4. Dashboard UI (port 8501):                                    ║
║     streamlit run ui/dashboard.py                                ║
╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
