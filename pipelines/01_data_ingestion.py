import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from config.utils import load_config, get_logger, audit_log, ensure_dirs

logger = get_logger("data_ingestion")

def generate_synthetic_pharma_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_samples
    data = {
        "mol_weight":          rng.normal(350, 80, n).clip(100, 700),
        "logp":                rng.normal(2.5, 1.2, n).clip(-2, 7),
        "tpsa":                rng.normal(75, 25, n).clip(20, 200),
        "hbd":                 rng.integers(0, 6, n).astype(float),
        "hba":                 rng.integers(0, 12, n).astype(float),
        "rotatable_bonds":     rng.integers(0, 15, n).astype(float),
        "aromatic_rings":      rng.integers(0, 5, n).astype(float),
        "patient_age":         rng.integers(18, 80, n).astype(float),
        "patient_weight_kg":   rng.normal(72, 15, n).clip(40, 150),
        "creatinine_clearance":rng.normal(90, 20, n).clip(15, 150),
        "is_male":             rng.integers(0, 2, n).astype(float),
        "baseline_crp":        rng.exponential(5, n).clip(0, 80),
        "baseline_il6":        rng.exponential(10, n).clip(0, 200),
        "baseline_tnfa":       rng.exponential(15, n).clip(0, 300),
        "baseline_wbc":        rng.normal(7, 2, n).clip(2, 20),
        "dose_mg":             rng.choice([25, 50, 100, 200, 400], n).astype(float),
        "treatment_days":      rng.integers(7, 90, n).astype(float),
    }

    df = pd.DataFrame(data)

    lipinski_ok = (
        (df["mol_weight"] <= 500) &
        (df["logp"] <= 5) &
        (df["hbd"] <= 5) &
        (df["hba"] <= 10)
    ).astype(float)

    score = (
        0.30 * lipinski_ok +
        0.15 * (df["dose_mg"] / 400) +
        0.10 * (df["treatment_days"] / 90) +
        0.10 * (1 - df["baseline_crp"] / 80) +
        0.10 * (df["creatinine_clearance"] / 150) +
        0.25 * rng.uniform(0, 1, n)
    )

    df["efficacy_label"] = (score > 0.45).astype(int)
    df["sample_id"] = [f"SAMPLE_{i:05d}" for i in range(n)]
    df["batch_id"]  = rng.choice(["BATCH_001", "BATCH_002", "BATCH_003"], n)
    df["ingestion_timestamp"] = pd.Timestamp.utcnow().isoformat()

    return df

def validate_schema(df: pd.DataFrame) -> bool:
    required_columns = [
        "mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds",
        "aromatic_rings", "patient_age", "patient_weight_kg", "creatinine_clearance",
        "is_male", "baseline_crp", "baseline_il6", "baseline_tnfa", "baseline_wbc",
        "dose_mg", "treatment_days", "efficacy_label", "sample_id", "batch_id"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error(f"Schema validation failed. Missing columns: {missing}")
        return False
    if df.isnull().sum().sum() > 0:
        logger.warning(f"Dataset has {df.isnull().sum().sum()} null values — filling with median")
    return True

def run():
    ensure_dirs()
    cfg = load_config()
    raw_path = Path(cfg["paths"]["raw_data"])

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 60)

    logger.info("Generating synthetic pharma training dataset (1000 samples)...")
    df_train = generate_synthetic_pharma_data(n_samples=1000, seed=42)

    logger.info("Generating synthetic production/inference batch (200 samples, shifted distribution)...")
    df_prod = generate_synthetic_pharma_data(n_samples=200, seed=999)
    df_prod["patient_age"]   = (df_prod["patient_age"] * 1.1).clip(18, 80)
    df_prod["baseline_crp"]  = (df_prod["baseline_crp"] * 1.4).clip(0, 80)

    logger.info("Validating schema...")
    if not validate_schema(df_train):
        raise RuntimeError("Schema validation failed for training data")

    train_path = raw_path / "drug_trials_train.csv"
    prod_path  = raw_path / "drug_trials_production.csv"
    df_train.to_csv(train_path, index=False)
    df_prod.to_csv(prod_path, index=False)

    logger.info(f"Saved {len(df_train)} training samples → {train_path}")
    logger.info(f"Saved {len(df_prod)} production samples → {prod_path}")

    try:
        from config.storage import upload_csv
        raw_bucket = cfg["minio"]["buckets"]["raw"]
        upload_csv(df_train, raw_bucket, "drug_trials_train.csv")
        upload_csv(df_prod,  raw_bucket, "drug_trials_production.csv")
        logger.info(f"Uploaded to MinIO bucket: {raw_bucket}")
    except Exception as e:
        logger.warning(f"MinIO upload failed (local files still saved): {e}")

    logger.info(f"Efficacy rate (train): {df_train['efficacy_label'].mean():.2%}")
    logger.info(f"Efficacy rate (prod):  {df_prod['efficacy_label'].mean():.2%}")

    audit_log("data_ingested", {
        "train_samples": len(df_train),
        "prod_samples": len(df_prod),
        "train_path": str(train_path),
        "prod_path": str(prod_path),
        "efficacy_rate_train": round(df_train["efficacy_label"].mean(), 4),
        "columns": list(df_train.columns),
    })

    logger.info("Stage 1 complete.\n")
    return df_train, df_prod

if __name__ == "__main__":
    run()