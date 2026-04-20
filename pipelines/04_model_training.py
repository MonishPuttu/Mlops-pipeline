# pipelines/04_model_training.py
# Trains models and logs everything to MLflow (experiments, metrics, artifacts)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from config.utils import load_config, get_logger, audit_log

logger = get_logger("model_training")

# Feature columns used for training (exclude metadata & target)
FEATURE_COLS = [
    "mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "aromatic_rings",
    "patient_age", "patient_weight_kg", "creatinine_clearance", "is_male",
    "baseline_crp", "baseline_il6", "baseline_tnfa", "baseline_wbc",
    "dose_mg", "treatment_days",
    # Engineered features
    "ro5_violations", "is_drug_like", "bioavailability_score", "inflammation_score",
    "renal_adjusted_dose", "bsa_proxy", "dose_per_bsa", "treatment_intensity",
    "age_metabolism_factor", "log_baseline_crp", "log_baseline_il6", "log_baseline_tnfa",
]

TARGET_COL = "efficacy_label"


def get_candidate_models():
    """Returns candidate model pipelines to compare."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]),
    }


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def run():
    cfg = load_config()
    proc_path  = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["models"])
    model_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 4: MODEL TRAINING")
    logger.info("=" * 60)

    # Load feature data
    feat_file = proc_path / "features_train.csv"
    if not feat_file.exists():
        raise FileNotFoundError(f"Run stage 3 first. Missing: {feat_file}")

    df = pd.read_csv(feat_file)

    # Filter to available feature columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        logger.warning(f"Missing features (will skip): {missing_features}")

    X = df[available_features].fillna(df[available_features].median())
    y = df[TARGET_COL]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(available_features)}")
    logger.info(f"Efficacy rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    best_model = None
    best_f1    = 0
    best_run_id = None
    all_results = []

    candidates = get_candidate_models()

    for model_name, pipeline in candidates.items():
        logger.info(f"\nTraining: {model_name}...")

        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("n_features", len(available_features))
            mlflow.log_param("feature_list", ",".join(available_features))

            # Train
            pipeline.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cfg["model"]["cv_folds"], scoring="f1"
            )
            mlflow.log_metric("cv_f1_mean", round(cv_scores.mean(), 4))
            mlflow.log_metric("cv_f1_std",  round(cv_scores.std(), 4))

            # Test metrics
            y_pred = pipeline.predict(X_test)
            y_prob = None
            if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Feature importances (if tree-based)
            clf = pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": available_features,
                    "importance": clf.feature_importances_
                }).sort_values("importance", ascending=False)
                fi_path = model_path / f"feature_importance_{model_name}.csv"
                fi.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))

            # Log model
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            run_id = run.info.run_id
            logger.info(f"  Metrics: acc={metrics['accuracy']} | prec={metrics['precision']} | rec={metrics['recall']} | f1={metrics['f1_score']}")
            logger.info(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"  MLflow run_id: {run_id}")

            all_results.append({
                "model_name": model_name,
                "run_id": run_id,
                **metrics,
                "cv_f1_mean": round(cv_scores.mean(), 4),
            })

            # Track best model
            if metrics["f1_score"] > best_f1:
                best_f1     = metrics["f1_score"]
                best_model  = pipeline
                best_run_id = run_id
                best_name   = model_name

    # Save best model locally
    best_model_path = model_path / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    # Save feature list for serving
    import json
    meta_path = model_path / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model_name": best_name,
            "feature_columns": available_features,
            "target_column": TARGET_COL,
            "best_run_id": best_run_id,
            "best_f1": best_f1,
            "trained_at": datetime.utcnow().isoformat(),
        }, f, indent=2)

    # Print comparison table
    results_df = pd.DataFrame(all_results)
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(results_df[["model_name","accuracy","precision","recall","f1_score","roc_auc","cv_f1_mean"]].to_string(index=False))
    logger.info(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    logger.info(f"Saved to: {best_model_path}")

    audit_log("model_trained", {
        "best_model": best_name,
        "best_f1": best_f1,
        "best_run_id": best_run_id,
        "models_compared": len(candidates),
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "n_features": len(available_features),
        "all_results": all_results,
    })

    logger.info("Stage 4 complete.\n")
    return best_model, available_features, best_run_id

if __name__ == "__main__":
    run()