# pipelines/02_data_validation.py
# Implements pharma-grade data quality checks (IQ - Installation Qualification equivalent)
# Checks ranges, distributions, null rates, statistical anomalies

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from config.utils import load_config, get_logger, audit_log

logger = get_logger("data_validation")


class PharmaDataValidator:
    """
    Simulates Great Expectations validation suite.
    Enforces pharma-specific data contracts and ranges.
    """

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def _check(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        entry = {"check": name, "status": status, "details": details}
        self.results.append(entry)
        icon = "✓" if condition else "✗"
        logger.info(f"  [{icon}] {name}: {details}")
        return condition

    def validate(self, df: pd.DataFrame) -> dict:
        logger.info(f"Running {self.__class__.__name__} on {len(df)} rows...")

        # --- Completeness checks ---
        self._check(
            "no_null_sample_ids",
            df["sample_id"].notna().all(),
            f"nulls={df['sample_id'].isna().sum()}"
        )
        null_rate = df.isnull().mean().mean()
        self._check(
            "overall_null_rate_below_5pct",
            null_rate < 0.05,
            f"null_rate={null_rate:.2%}"
        )

        # --- Domain range checks (Lipinski / clinical) ---
        self._check(
            "mol_weight_range",
            df["mol_weight"].between(50, 1000).all(),
            f"range=[{df['mol_weight'].min():.1f}, {df['mol_weight'].max():.1f}]"
        )
        self._check(
            "logp_range",
            df["logp"].between(-5, 10).all(),
            f"range=[{df['logp'].min():.2f}, {df['logp'].max():.2f}]"
        )
        self._check(
            "patient_age_range",
            df["patient_age"].between(0, 120).all(),
            f"range=[{df['patient_age'].min():.0f}, {df['patient_age'].max():.0f}]"
        )
        self._check(
            "dose_mg_valid_values",
            df["dose_mg"].isin([25, 50, 100, 200, 400]).all(),
            f"unique_values={sorted(df['dose_mg'].unique())}"
        )
        self._check(
            "efficacy_label_binary",
            df["efficacy_label"].isin([0, 1]).all(),
            f"unique_values={sorted(df['efficacy_label'].unique())}"
        )

        # --- Statistical checks ---
        label_balance = df["efficacy_label"].mean()
        self._check(
            "label_balance_not_extreme",
            0.15 < label_balance < 0.85,
            f"efficacy_rate={label_balance:.2%}"
        )

        # Check for duplicate sample IDs
        dup_count = df["sample_id"].duplicated().sum()
        self._check(
            "no_duplicate_sample_ids",
            dup_count == 0,
            f"duplicates={dup_count}"
        )

        # Minimum sample size for reliable ML
        self._check(
            "minimum_sample_size",
            len(df) >= 100,
            f"n_samples={len(df)}"
        )

        # Check baseline biomarkers are non-negative
        for col in ["baseline_crp", "baseline_il6", "baseline_tnfa", "baseline_wbc"]:
            self._check(
                f"{col}_non_negative",
                (df[col] >= 0).all(),
                f"min={df[col].min():.3f}"
            )

        # Treatment days sanity
        self._check(
            "treatment_days_positive",
            (df["treatment_days"] > 0).all(),
            f"range=[{df['treatment_days'].min():.0f}, {df['treatment_days'].max():.0f}]"
        )

        summary = {
            "total_checks": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.passed / (self.passed + self.failed),
            "results": self.results,
        }
        return summary


def run():
    cfg = load_config()
    raw_path = Path(cfg["paths"]["raw_data"])

    logger.info("=" * 60)
    logger.info("STAGE 2: DATA VALIDATION")
    logger.info("=" * 60)

    # Load training data
    train_file = raw_path / "drug_trials_train.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"Run 01_data_ingestion.py first. Missing: {train_file}")

    df = pd.read_csv(train_file)
    logger.info(f"Loaded {len(df)} rows from {train_file}")

    # Run validation
    validator = PharmaDataValidator()
    summary = validator.validate(df)

    # Save validation report
    processed_path = Path(cfg["paths"]["processed_data"])
    processed_path.mkdir(parents=True, exist_ok=True)
    report_path = processed_path / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nValidation Summary: {summary['passed']}/{summary['total_checks']} checks passed")

    if summary["failed"] > 0:
        failed_checks = [r for r in summary["results"] if r["status"] == "FAIL"]
        logger.warning(f"Failed checks: {[r['check'] for r in failed_checks]}")

    audit_log("data_validated", {
        "input_file": str(train_file),
        "total_checks": summary["total_checks"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "pass_rate": round(summary["pass_rate"], 4),
    })

    # Pipeline gate: block if critical failures
    if summary["failed"] > 3:
        raise RuntimeError(f"Too many validation failures ({summary['failed']}). Pipeline blocked.")

    logger.info("Stage 2 complete.\n")
    return df, summary

if __name__ == "__main__":
    run()
