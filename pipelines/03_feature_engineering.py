import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
from config.utils import load_config, get_logger, audit_log

logger = get_logger("feature_engineering")


class LocalFeatureStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_set_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    n_samples INTEGER,
                    n_features INTEGER,
                    feature_names TEXT,
                    description TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_data (
                    sample_id TEXT NOT NULL,
                    feature_set_version TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (sample_id, feature_set_version)
                )
            """)

    def save_feature_set(self, name: str, version: str, df: pd.DataFrame, description: str = ""):
        ts = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_sets
                (feature_set_name, version, created_at, n_samples, n_features, feature_names, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, version, ts, len(df), len(df.columns),
                  ",".join(df.columns.tolist()), description))
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO feature_data
                    (sample_id, feature_set_version, features_json, created_at)
                    VALUES (?, ?, ?, ?)
                """, (row.get("sample_id", "UNKNOWN"), version, row.to_json(), ts))
        logger.info(f"Saved feature set '{name}' v{version}: {len(df)} samples, {len(df.columns)} features")

    def get_feature_sets(self):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("SELECT * FROM feature_sets ORDER BY created_at DESC", conn)

    def get_latest_version(self, name: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT version FROM feature_sets
                WHERE feature_set_name = ?
                ORDER BY created_at DESC LIMIT 1
            """, (name,)).fetchone()
        return row[0] if row else None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    feat["ro5_mw_ok"]  = (feat["mol_weight"] <= 500).astype(int)
    feat["ro5_logp_ok"]= (feat["logp"] <= 5).astype(int)
    feat["ro5_hbd_ok"] = (feat["hbd"] <= 5).astype(int)
    feat["ro5_hba_ok"] = (feat["hba"] <= 10).astype(int)
    feat["ro5_violations"] = 4 - (
        feat["ro5_mw_ok"] + feat["ro5_logp_ok"] +
        feat["ro5_hbd_ok"] + feat["ro5_hba_ok"]
    )
    feat["is_drug_like"] = (feat["ro5_violations"] <= 1).astype(int)

    feat["bioavailability_score"] = (
        (1 - feat["ro5_violations"] / 4) * 0.5 +
        ((feat["tpsa"] < 140).astype(float)) * 0.3 +
        ((feat["rotatable_bonds"] < 10).astype(float)) * 0.2
    ).clip(0, 1)

    feat["inflammation_score"] = (
        feat["baseline_crp"]  / feat["baseline_crp"].max() * 0.4 +
        feat["baseline_il6"]  / feat["baseline_il6"].max() * 0.35 +
        feat["baseline_tnfa"] / feat["baseline_tnfa"].max() * 0.25
    ).clip(0, 1)

    feat["renal_adjusted_dose"] = (
        feat["dose_mg"] * (feat["creatinine_clearance"] / 90)
    ).clip(0, 800)

    feat["bsa_proxy"] = (feat["patient_weight_kg"] / 70) ** 0.5
    feat["dose_per_bsa"] = feat["dose_mg"] / feat["bsa_proxy"]
    feat["treatment_intensity"] = feat["dose_mg"] * feat["treatment_days"] / 1000

    feat["age_metabolism_factor"] = np.where(
        feat["patient_age"] < 30, 1.2,
        np.where(feat["patient_age"] > 65, 0.7, 1.0)
    )

    for col in ["baseline_crp", "baseline_il6", "baseline_tnfa"]:
        feat[f"log_{col}"] = np.log1p(feat[col])

    feat = feat.drop(columns=["ingestion_timestamp"], errors="ignore")

    return feat


def run():
    cfg = load_config()
    raw_path  = Path(cfg["paths"]["raw_data"])
    proc_path = Path(cfg["paths"]["processed_data"])
    feat_path = Path(cfg["paths"]["features"])
    proc_path.mkdir(parents=True, exist_ok=True)
    feat_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    train_file = raw_path / "drug_trials_train.csv"
    df = pd.read_csv(train_file)
    logger.info(f"Loaded {len(df)} training samples")

    logger.info("Engineering pharma features...")
    df_feat = engineer_features(df)

    n_new_features = len(df_feat.columns) - len(df.columns)
    logger.info(f"Created {n_new_features} new features. Total columns: {len(df_feat.columns)}")

    processed_file = proc_path / "features_train.csv"
    df_feat.to_csv(processed_file, index=False)

    reference_file = proc_path / "reference.csv"
    df_feat.to_csv(reference_file, index=False)
    logger.info(f"Reference dataset saved → {reference_file}")

    prod_raw = raw_path / "drug_trials_production.csv"
    if prod_raw.exists():
        df_prod = pd.read_csv(prod_raw)
        df_prod_feat = engineer_features(df_prod)
        prod_feat_file = proc_path / "features_production.csv"
        df_prod_feat.to_csv(prod_feat_file, index=False)
        logger.info(f"Production features saved → {prod_feat_file}")

    try:
        from config.storage import upload_csv
        proc_bucket = cfg["minio"]["buckets"]["processed"]
        upload_csv(df_feat,      proc_bucket, "features_train.csv")
        upload_csv(df_feat,      proc_bucket, "reference.csv")
        if prod_raw.exists():
            upload_csv(df_prod_feat, proc_bucket, "features_production.csv")
        logger.info(f"Uploaded processed features to MinIO bucket: {proc_bucket}")
    except Exception as e:
        logger.warning(f"MinIO upload failed (local files still saved): {e}")

    store = LocalFeatureStore(str(feat_path / "feature_store.db"))
    store.save_feature_set(
        name="drug_efficacy_features",
        version="v1.0",
        df=df_feat,
        description="Pharma drug trial features with Lipinski, inflammation, and renal scores"
    )

    audit_log("features_engineered", {
        "input_file": str(train_file),
        "output_file": str(processed_file),
        "n_samples": len(df_feat),
        "n_original_features": len(df.columns),
        "n_engineered_features": n_new_features,
        "total_features": len(df_feat.columns),
        "feature_store_version": "v1.0",
    })

    logger.info("Stage 3 complete.\n")
    return df_feat

if __name__ == "__main__":
    run()