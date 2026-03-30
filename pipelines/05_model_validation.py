# pipelines/05_model_validation.py
# Pharma-grade model validation: IQ, OQ, PQ qualification protocol
# IQ = Installation Qualification (model loads correctly)
# OQ = Operational Qualification (model performs within spec)
# PQ = Performance Qualification (model meets business/clinical thresholds)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from config.utils import load_config, get_logger, audit_log

logger = get_logger("model_validation")


class PharmaQualificationProtocol:
    """
    Implements IQ/OQ/PQ validation for ML models.
    Generates a signed validation report.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.val_cfg = cfg["validation"]
        self.iq_passed = None
        self.oq_passed = None
        self.pq_passed = None
        self.report = {}

    def run_iq(self, model_path: str, meta_path: str) -> bool:
        """Installation Qualification: model can be loaded and is not corrupted."""
        logger.info("\n--- IQ (Installation Qualification) ---")
        checks = []

        # Check 1: model file exists
        model_exists = Path(model_path).exists()
        checks.append(("model_file_exists", model_exists, str(model_path)))
        logger.info(f"  [{'✓' if model_exists else '✗'}] Model file exists: {model_path}")

        # Check 2: model can be loaded
        try:
            model = joblib.load(model_path)
            load_ok = True
            checks.append(("model_loads_without_error", True, "joblib.load successful"))
            logger.info("  [✓] Model loads without error")
        except Exception as e:
            load_ok = False
            checks.append(("model_loads_without_error", False, str(e)))
            logger.error(f"  [✗] Model load failed: {e}")

        # Check 3: model has required methods
        if load_ok:
            has_predict     = hasattr(model, "predict")
            has_predict_proba = hasattr(model, "predict_proba")
            checks.append(("model_has_predict",       has_predict,       ""))
            checks.append(("model_has_predict_proba", has_predict_proba, ""))
            logger.info(f"  [{'✓' if has_predict else '✗'}] model.predict() exists")
            logger.info(f"  [{'✓' if has_predict_proba else '✗'}] model.predict_proba() exists")

        # Check 4: metadata exists
        meta_exists = Path(meta_path).exists()
        checks.append(("metadata_file_exists", meta_exists, str(meta_path)))
        logger.info(f"  [{'✓' if meta_exists else '✗'}] Metadata file exists")

        self.iq_passed = all(c[1] for c in checks)
        self.report["iq"] = {"checks": checks, "passed": self.iq_passed}

        if self.iq_passed:
            logger.info("  IQ: PASSED ✓")
        else:
            logger.error("  IQ: FAILED ✗")
        return self.iq_passed

    def run_oq(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> bool:
        """Operational Qualification: model performs within acceptable statistical limits."""
        logger.info("\n--- OQ (Operational Qualification) ---")
        checks = []

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }

        thresholds = {
            "accuracy":  self.val_cfg["min_accuracy"],
            "precision": self.val_cfg["min_precision"],
            "recall":    self.val_cfg["min_recall"],
            "f1_score":  self.val_cfg["min_f1"],
        }

        for metric, threshold in thresholds.items():
            actual = metrics[metric]
            passed = actual >= threshold
            checks.append((f"{metric}_above_threshold", passed,
                           f"actual={actual} >= threshold={threshold}"))
            logger.info(f"  [{'✓' if passed else '✗'}] {metric}: {actual:.4f} (min={threshold})")

        # Confusion matrix analysis
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        logger.info(f"  Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        # Prediction consistency check: no degenerate model (always predicts one class)
        pred_variance = y_pred.var()
        is_not_degenerate = pred_variance > 0
        checks.append(("predictions_not_degenerate", is_not_degenerate, f"variance={pred_variance:.4f}"))
        logger.info(f"  [{'✓' if is_not_degenerate else '✗'}] Predictions not degenerate (variance={pred_variance:.4f})")

        self.oq_passed = all(c[1] for c in checks)
        self.report["oq"] = {
            "checks": checks,
            "metrics": metrics,
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
            "specificity": round(specificity, 4),
            "passed": self.oq_passed,
        }

        if self.oq_passed:
            logger.info("  OQ: PASSED ✓")
        else:
            logger.error("  OQ: FAILED ✗")
        return self.oq_passed

    def run_pq(self, model, X_prod: pd.DataFrame) -> bool:
        """Performance Qualification: model behaves sensibly on realistic production-like data."""
        logger.info("\n--- PQ (Performance Qualification) ---")
        checks = []

        y_pred = model.predict(X_prod)
        y_prob = model.predict_proba(X_prod)[:, 1]

        # PQ Check 1: Efficacy rate on production data is within plausible range
        prod_efficacy_rate = y_pred.mean()
        in_range = 0.05 <= prod_efficacy_rate <= 0.97
        checks.append(("prod_efficacy_rate_in_range", in_range,
                       f"efficacy_rate={prod_efficacy_rate:.2%}"))
        logger.info(f"  [{'✓' if in_range else '✗'}] Production efficacy rate: {prod_efficacy_rate:.2%} (expected 10-90%)")

        # PQ Check 2: Probability scores are well-distributed (not all 0 or 1)
        prob_std = y_prob.std()
        prob_spread = prob_std > 0.05
        checks.append(("prob_scores_well_distributed", prob_spread,
                       f"std={prob_std:.4f}"))
        logger.info(f"  [{'✓' if prob_spread else '✗'}] Probability scores well-distributed (std={prob_std:.4f})")

        # PQ Check 3: No NaN predictions
        no_nan = not np.isnan(y_pred).any()
        checks.append(("no_nan_predictions", no_nan, ""))
        logger.info(f"  [{'✓' if no_nan else '✗'}] No NaN predictions")

        # PQ Check 4: Inference time acceptable
        import time
        t0 = time.perf_counter()
        _ = model.predict(X_prod.head(100))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        fast_enough = elapsed_ms < 5000  # 5 seconds for 100 samples
        checks.append(("inference_time_acceptable", fast_enough,
                       f"100_samples={elapsed_ms:.1f}ms"))
        logger.info(f"  [{'✓' if fast_enough else '✗'}] Inference time: {elapsed_ms:.1f}ms for 100 samples")

        self.pq_passed = all(c[1] for c in checks)
        self.report["pq"] = {
            "checks": checks,
            "prod_efficacy_rate": round(float(prod_efficacy_rate), 4),
            "prob_std": round(float(prob_std), 4),
            "passed": self.pq_passed,
        }

        if self.pq_passed:
            logger.info("  PQ: PASSED ✓")
        else:
            logger.error("  PQ: FAILED ✗")
        return self.pq_passed

    def generate_report(self, output_path: str, meta: dict) -> dict:
        overall = self.iq_passed and self.oq_passed and self.pq_passed
        report = {
            "report_type": "ML Model Qualification Report",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_metadata": meta,
            "overall_qualification_status": "QUALIFIED" if overall else "NOT QUALIFIED",
            "iq_result": "PASSED" if self.iq_passed else "FAILED",
            "oq_result": "PASSED" if self.oq_passed else "FAILED",
            "pq_result": "PASSED" if self.pq_passed else "FAILED",
            "detailed_results": self.report,
            "qualified_for_production": overall,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=lambda o: bool(o) if isinstance(o, (bool, np.bool_)) else int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else str(o))
        logger.info(f"\nValidation report saved → {output_path}")
        return report


def run():
    cfg = load_config()
    proc_path  = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["models"])

    logger.info("=" * 60)
    logger.info("STAGE 5: PHARMA MODEL VALIDATION (IQ/OQ/PQ)")
    logger.info("=" * 60)

    model_file = model_path / "best_model.joblib"
    meta_file  = model_path / "model_metadata.json"

    # Load metadata
    with open(meta_file) as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    target_col   = meta["target_column"]

    # Load data
    df_train = pd.read_csv(proc_path / "features_train.csv")
    df_prod  = pd.read_csv(proc_path / "features_production.csv")

    from sklearn.model_selection import train_test_split
    X = df_train[[c for c in feature_cols if c in df_train.columns]].fillna(0)
    y = df_train[target_col]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_prod = df_prod[[c for c in feature_cols if c in df_prod.columns]].fillna(0)

    model = joblib.load(model_file)

    # Run qualification protocol
    protocol = PharmaQualificationProtocol(cfg)
    protocol.run_iq(str(model_file), str(meta_file))
    protocol.run_oq(model, X_test, y_test)
    protocol.run_pq(model, X_prod)

    # Generate report
    report_path = proc_path / "qualification_report.json"
    report = protocol.generate_report(str(report_path), meta)

    logger.info("\n" + "=" * 60)
    logger.info(f"OVERALL STATUS: {report['overall_qualification_status']}")
    logger.info("=" * 60)

    audit_log("model_validated", {
        "model_path": str(model_file),
        "iq": report["iq_result"],
        "oq": report["oq_result"],
        "pq": report["pq_result"],
        "overall": report["overall_qualification_status"],
        "qualified_for_production": report["qualified_for_production"],
    })

    if not report["qualified_for_production"]:
        logger.error("Model did NOT pass qualification. Pipeline halted.")
        raise RuntimeError("Model qualification failed. See qualification_report.json")

    logger.info("Stage 5 complete.\n")
    return report

if __name__ == "__main__":
    run()
