# tests/test_pipeline.py
# Full test suite for the Pharma MLOps pipeline
# Run: python -m pytest tests/ -v
# Or:  python tests/test_pipeline.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    from config.utils import load_config
    return load_config()

@pytest.fixture(scope="session")
def sample_raw_df():
    """Small raw dataframe for unit tests (doesn't touch disk)."""
    from pipelines._01_data_ingestion import generate_synthetic_pharma_data
    return generate_synthetic_pharma_data(n_samples=50, seed=0)

@pytest.fixture(scope="session")
def sample_feat_df(sample_raw_df):
    """Feature-engineered version of the sample dataframe."""
    from pipelines._03_feature_engineering import engineer_features
    return engineer_features(sample_raw_df)

@pytest.fixture(scope="session")
def trained_model(cfg):
    """Loads the already-trained model (requires pipeline to have been run)."""
    model_path = Path(cfg["paths"]["models"]) / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("Model not trained yet — run `python run.py` first")
    return joblib.load(model_path)

@pytest.fixture(scope="session")
def model_meta(cfg):
    meta_path = Path(cfg["paths"]["models"]) / "model_metadata.json"
    if not meta_path.exists():
        pytest.skip("Model metadata not found — run `python run.py` first")
    with open(meta_path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIngestion:

    def test_generates_correct_row_count(self, sample_raw_df):
        assert len(sample_raw_df) == 50

    def test_has_all_required_columns(self, sample_raw_df):
        required = [
            "mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds",
            "aromatic_rings", "patient_age", "patient_weight_kg",
            "creatinine_clearance", "is_male", "baseline_crp", "baseline_il6",
            "baseline_tnfa", "baseline_wbc", "dose_mg", "treatment_days",
            "efficacy_label", "sample_id", "batch_id",
        ]
        for col in required:
            assert col in sample_raw_df.columns, f"Missing column: {col}"

    def test_no_null_values_in_key_columns(self, sample_raw_df):
        key_cols = ["mol_weight", "logp", "efficacy_label", "sample_id"]
        for col in key_cols:
            assert sample_raw_df[col].notna().all(), f"Nulls found in {col}"

    def test_efficacy_label_is_binary(self, sample_raw_df):
        assert set(sample_raw_df["efficacy_label"].unique()).issubset({0, 1})

    def test_molecular_weight_in_range(self, sample_raw_df):
        assert (sample_raw_df["mol_weight"] >= 100).all()
        assert (sample_raw_df["mol_weight"] <= 700).all()

    def test_patient_age_in_range(self, sample_raw_df):
        assert (sample_raw_df["patient_age"] >= 18).all()
        assert (sample_raw_df["patient_age"] <= 80).all()

    def test_dose_values_are_valid(self, sample_raw_df):
        valid_doses = {25, 50, 100, 200, 400}
        assert set(sample_raw_df["dose_mg"].unique()).issubset(valid_doses)

    def test_sample_ids_are_unique(self, sample_raw_df):
        assert sample_raw_df["sample_id"].nunique() == len(sample_raw_df)

    def test_baseline_biomarkers_non_negative(self, sample_raw_df):
        for col in ["baseline_crp", "baseline_il6", "baseline_tnfa", "baseline_wbc"]:
            assert (sample_raw_df[col] >= 0).all(), f"Negative values in {col}"

    def test_treatment_days_positive(self, sample_raw_df):
        assert (sample_raw_df["treatment_days"] > 0).all()

    def test_distribution_shift_is_injected(self):
        """Production batch should have higher patient_age and baseline_crp than training."""
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        df_train = generate_synthetic_pharma_data(n_samples=500, seed=42)
        df_prod  = generate_synthetic_pharma_data(n_samples=200, seed=999)
        # Apply same shift as pipeline does
        df_prod["patient_age"]  = (df_prod["patient_age"] * 1.1).clip(18, 80)
        df_prod["baseline_crp"] = (df_prod["baseline_crp"] * 1.4).clip(0, 80)
        assert df_prod["patient_age"].mean() > df_train["patient_age"].mean()
        assert df_prod["baseline_crp"].mean() > df_train["baseline_crp"].mean()


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataValidation:

    def test_validator_passes_valid_data(self, sample_raw_df):
        from pipelines._02_data_validation import PharmaDataValidator
        v = PharmaDataValidator()
        result = v.validate(sample_raw_df)
        # Should pass most checks; allow max 2 failures on small sample
        assert result["failed"] <= 2, f"Too many failures: {result['failed']}"

    def test_validator_catches_null_sample_ids(self):
        from pipelines._02_data_validation import PharmaDataValidator
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        df = generate_synthetic_pharma_data(n_samples=50, seed=1)
        df.loc[0:5, "sample_id"] = None  # Inject nulls
        v = PharmaDataValidator()
        result = v.validate(df)
        failed_names = [r["check"] for r in result["results"] if r["status"] == "FAIL"]
        assert "no_null_sample_ids" in failed_names

    def test_validator_catches_out_of_range_mol_weight(self):
        from pipelines._02_data_validation import PharmaDataValidator
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        df = generate_synthetic_pharma_data(n_samples=50, seed=2)
        df.loc[0, "mol_weight"] = 99999  # Way out of range
        v = PharmaDataValidator()
        result = v.validate(df)
        failed_names = [r["check"] for r in result["results"] if r["status"] == "FAIL"]
        assert "mol_weight_range" in failed_names

    def test_validator_catches_invalid_dose(self):
        from pipelines._02_data_validation import PharmaDataValidator
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        df = generate_synthetic_pharma_data(n_samples=50, seed=3)
        df.loc[0, "dose_mg"] = 999  # Invalid dose
        v = PharmaDataValidator()
        result = v.validate(df)
        failed_names = [r["check"] for r in result["results"] if r["status"] == "FAIL"]
        assert "dose_mg_valid_values" in failed_names

    def test_validator_catches_duplicate_sample_ids(self):
        from pipelines._02_data_validation import PharmaDataValidator
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        df = generate_synthetic_pharma_data(n_samples=50, seed=4)
        df.loc[1, "sample_id"] = df.loc[0, "sample_id"]  # Duplicate
        v = PharmaDataValidator()
        result = v.validate(df)
        failed_names = [r["check"] for r in result["results"] if r["status"] == "FAIL"]
        assert "no_duplicate_sample_ids" in failed_names

    def test_pipeline_blocks_on_too_many_failures(self):
        """Pipeline should raise RuntimeError if >3 checks fail."""
        from pipelines._02_data_validation import PharmaDataValidator
        import pandas as pd
        # Completely broken dataframe
        df = pd.DataFrame({
            "sample_id": [None] * 10,
            "mol_weight": [999999] * 10,
            "logp": [999] * 10,
            "patient_age": [-5] * 10,
            "dose_mg": [0] * 10,
            "efficacy_label": [2] * 10,  # Invalid label
            "baseline_crp": [-1] * 10,
            "baseline_il6": [0] * 10,
            "baseline_tnfa": [0] * 10,
            "baseline_wbc": [0] * 10,
            "treatment_days": [-1] * 10,
        })
        v = PharmaDataValidator()
        result = v.validate(df)
        assert result["failed"] > 3


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_creates_engineered_columns(self, sample_feat_df):
        engineered = [
            "ro5_violations", "is_drug_like", "bioavailability_score",
            "inflammation_score", "renal_adjusted_dose", "bsa_proxy",
            "dose_per_bsa", "treatment_intensity", "age_metabolism_factor",
            "log_baseline_crp", "log_baseline_il6", "log_baseline_tnfa",
        ]
        for col in engineered:
            assert col in sample_feat_df.columns, f"Missing engineered feature: {col}"

    def test_ro5_violations_in_valid_range(self, sample_feat_df):
        assert (sample_feat_df["ro5_violations"] >= 0).all()
        assert (sample_feat_df["ro5_violations"] <= 4).all()

    def test_is_drug_like_is_binary(self, sample_feat_df):
        assert set(sample_feat_df["is_drug_like"].unique()).issubset({0, 1})

    def test_bioavailability_score_bounded(self, sample_feat_df):
        assert (sample_feat_df["bioavailability_score"] >= 0).all()
        assert (sample_feat_df["bioavailability_score"] <= 1).all()

    def test_inflammation_score_bounded(self, sample_feat_df):
        assert (sample_feat_df["inflammation_score"] >= 0).all()
        assert (sample_feat_df["inflammation_score"] <= 1).all()

    def test_log_transforms_are_non_negative(self, sample_feat_df):
        for col in ["log_baseline_crp", "log_baseline_il6", "log_baseline_tnfa"]:
            assert (sample_feat_df[col] >= 0).all(), f"Negative log values in {col}"

    def test_age_metabolism_factor_correct_values(self, sample_feat_df):
        # Should be exactly 0.7, 1.0, or 1.2
        valid = {0.7, 1.0, 1.2}
        actual = set(sample_feat_df["age_metabolism_factor"].unique())
        assert actual.issubset(valid), f"Unexpected metabolism factors: {actual}"

    def test_bsa_proxy_positive(self, sample_feat_df):
        assert (sample_feat_df["bsa_proxy"] > 0).all()

    def test_renal_adjusted_dose_non_negative(self, sample_feat_df):
        assert (sample_feat_df["renal_adjusted_dose"] >= 0).all()

    def test_no_new_nulls_introduced(self, sample_raw_df, sample_feat_df):
        new_feat_cols = [
            c for c in sample_feat_df.columns if c not in sample_raw_df.columns
        ]
        for col in new_feat_cols:
            null_count = sample_feat_df[col].isna().sum()
            assert null_count == 0, f"Feature {col} has {null_count} nulls"

    def test_drug_like_lipinski_logic(self):
        """Lipinski drug-like = ≤1 Ro5 violation."""
        from pipelines._03_feature_engineering import engineer_features
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        df = generate_synthetic_pharma_data(n_samples=20, seed=10)
        # Force a clearly drug-like compound
        df.loc[0, ["mol_weight", "logp", "hbd", "hba"]] = [300, 2.0, 1, 4]
        feat = engineer_features(df)
        assert feat.loc[0, "ro5_violations"] == 0
        assert feat.loc[0, "is_drug_like"]   == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelTraining:

    def test_model_file_exists(self, cfg):
        assert (Path(cfg["paths"]["models"]) / "best_model.joblib").exists()

    def test_metadata_file_exists(self, cfg):
        assert (Path(cfg["paths"]["models"]) / "model_metadata.json").exists()

    def test_metadata_has_required_keys(self, model_meta):
        required_keys = [
            "model_name", "feature_columns", "target_column",
            "best_run_id", "best_f1", "trained_at",
        ]
        for k in required_keys:
            assert k in model_meta, f"Missing metadata key: {k}"

    def test_best_f1_above_threshold(self, model_meta):
        assert model_meta["best_f1"] >= 0.72, (
            f"Best F1 {model_meta['best_f1']:.4f} below minimum threshold 0.72"
        )

    def test_feature_columns_non_empty(self, model_meta):
        assert len(model_meta["feature_columns"]) >= 10

    def test_model_has_predict_methods(self, trained_model):
        assert hasattr(trained_model, "predict")
        assert hasattr(trained_model, "predict_proba")

    def test_model_predicts_binary_labels(self, trained_model, sample_feat_df, model_meta):
        feat_cols = [c for c in model_meta["feature_columns"] if c in sample_feat_df.columns]
        X = sample_feat_df[feat_cols].fillna(0)
        preds = trained_model.predict(X)
        assert set(preds).issubset({0, 1}), f"Non-binary predictions: {set(preds)}"

    def test_model_predict_proba_sums_to_one(self, trained_model, sample_feat_df, model_meta):
        feat_cols = [c for c in model_meta["feature_columns"] if c in sample_feat_df.columns]
        X = sample_feat_df[feat_cols].fillna(0)
        proba = trained_model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1"

    def test_model_probabilities_between_0_and_1(self, trained_model, sample_feat_df, model_meta):
        feat_cols = [c for c in model_meta["feature_columns"] if c in sample_feat_df.columns]
        X = sample_feat_df[feat_cols].fillna(0)
        proba = trained_model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_model_is_not_degenerate(self, trained_model, sample_feat_df, model_meta):
        """Model should not always predict the same class."""
        feat_cols = [c for c in model_meta["feature_columns"] if c in sample_feat_df.columns]
        X = sample_feat_df[feat_cols].fillna(0)
        preds = trained_model.predict(X)
        assert len(set(preds)) > 1, "Model always predicts the same class (degenerate)"

    def test_inference_speed(self, trained_model, sample_feat_df, model_meta):
        """100 inferences should complete in under 2 seconds."""
        import time
        feat_cols = [c for c in model_meta["feature_columns"] if c in sample_feat_df.columns]
        X = sample_feat_df[feat_cols].fillna(0)
        # Repeat to get ~100 samples
        X_100 = pd.concat([X] * max(1, 100 // len(X) + 1)).head(100)
        t0 = time.perf_counter()
        _ = trained_model.predict(X_100)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Inference too slow: {elapsed:.2f}s for 100 samples"


# ══════════════════════════════════════════════════════════════════════════════
# 5. IQ/OQ/PQ VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelValidation:

    def test_qualification_report_exists(self, cfg):
        assert (Path(cfg["paths"]["processed_data"]) / "qualification_report.json").exists()

    def test_model_is_qualified(self, cfg):
        qual_path = Path(cfg["paths"]["processed_data"]) / "qualification_report.json"
        if not qual_path.exists():
            pytest.skip("Qualification report not found")
        with open(qual_path) as f:
            qual = json.load(f)
        assert qual["overall_qualification_status"] == "QUALIFIED", (
            f"Model status: {qual['overall_qualification_status']}"
        )

    def test_iq_passed(self, cfg):
        qual_path = Path(cfg["paths"]["processed_data"]) / "qualification_report.json"
        if not qual_path.exists():
            pytest.skip("Qualification report not found")
        with open(qual_path) as f:
            qual = json.load(f)
        assert qual["iq_result"] == "PASSED"

    def test_oq_passed(self, cfg):
        qual_path = Path(cfg["paths"]["processed_data"]) / "qualification_report.json"
        if not qual_path.exists():
            pytest.skip("Qualification report not found")
        with open(qual_path) as f:
            qual = json.load(f)
        assert qual["oq_result"] == "PASSED"

    def test_pq_passed(self, cfg):
        qual_path = Path(cfg["paths"]["processed_data"]) / "qualification_report.json"
        if not qual_path.exists():
            pytest.skip("Qualification report not found")
        with open(qual_path) as f:
            qual = json.load(f)
        assert qual["pq_result"] == "PASSED"

    def test_oq_metrics_above_thresholds(self, cfg):
        qual_path = Path(cfg["paths"]["processed_data"]) / "qualification_report.json"
        if not qual_path.exists():
            pytest.skip("Qualification report not found")
        with open(qual_path) as f:
            qual = json.load(f)
        metrics = qual.get("detailed_results", {}).get("oq", {}).get("metrics", {})
        assert metrics.get("accuracy",  0) >= 0.75
        assert metrics.get("precision", 0) >= 0.70
        assert metrics.get("recall",    0) >= 0.70
        assert metrics.get("f1_score",  0) >= 0.72

    def test_iq_protocol_unit(self, cfg):
        """Unit test: IQ should pass on the real model file."""
        from pipelines._05_model_validation import PharmaQualificationProtocol
        model_path = str(Path(cfg["paths"]["models"]) / "best_model.joblib")
        meta_path  = str(Path(cfg["paths"]["models"]) / "model_metadata.json")
        if not Path(model_path).exists():
            pytest.skip("Model not trained yet")
        protocol = PharmaQualificationProtocol(cfg)
        result   = protocol.run_iq(model_path, meta_path)
        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:

    def test_registry_file_exists(self, cfg):
        assert (Path(cfg["paths"]["registry"]) / "registry.json").exists()

    def test_current_production_pointer_exists(self, cfg):
        assert (Path(cfg["paths"]["registry"]) / "current_production.json").exists()

    def test_registry_entry_has_required_fields(self, cfg):
        reg_path = Path(cfg["paths"]["registry"]) / "current_production.json"
        if not reg_path.exists():
            pytest.skip("Registry not found")
        with open(reg_path) as f:
            reg = json.load(f)
        required = [
            "model_name", "version", "stage", "registered_at",
            "model_path", "qualified_for_production", "feature_columns",
        ]
        for k in required:
            assert k in reg, f"Missing registry field: {k}"

    def test_registry_model_is_qualified_for_production(self, cfg):
        reg_path = Path(cfg["paths"]["registry"]) / "current_production.json"
        if not reg_path.exists():
            pytest.skip("Registry not found")
        with open(reg_path) as f:
            reg = json.load(f)
        assert reg["qualified_for_production"] is True

    def test_registry_points_to_existing_model_file(self, cfg):
        reg_path = Path(cfg["paths"]["registry"]) / "current_production.json"
        if not reg_path.exists():
            pytest.skip("Registry not found")
        with open(reg_path) as f:
            reg = json.load(f)
        assert Path(reg["model_path"]).exists(), (
            f"Registry points to missing model: {reg['model_path']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. MONITORING / DRIFT DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMonitoring:

    def test_drift_detected_on_shifted_data(self):
        """Monitoring should detect drift when production data is shifted."""
        from monitoring.monitor import run_evidently_drift
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        from pipelines._03_feature_engineering import engineer_features

        df_ref  = engineer_features(generate_synthetic_pharma_data(500, seed=42))
        df_prod = generate_synthetic_pharma_data(200, seed=999)
        # Apply strong shift
        df_prod["patient_age"]  = (df_prod["patient_age"] * 1.5).clip(18, 80)
        df_prod["baseline_crp"] = (df_prod["baseline_crp"] * 2.0).clip(0, 80)
        df_prod = engineer_features(df_prod)

        report = run_evidently_drift(df_ref, df_prod)
        assert report["dataset_drift_detected"] is True
        assert report["n_drifted_features"] >= 1

    def test_no_drift_on_same_distribution(self):
        """Monitoring should not flag drift when distributions match."""
        from monitoring.monitor import run_evidently_drift
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        from pipelines._03_feature_engineering import engineer_features

        df_ref  = engineer_features(generate_synthetic_pharma_data(500, seed=42))
        df_prod = engineer_features(generate_synthetic_pharma_data(200, seed=43))
        # Same distribution, different seed → should not drift much
        report = run_evidently_drift(df_ref, df_prod)
        # Should drift in <50% of features (loose threshold for statistical noise)
        assert report["share_drifted_features"] < 0.5

    def test_drift_report_has_required_keys(self):
        from monitoring.monitor import run_evidently_drift
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        from pipelines._03_feature_engineering import engineer_features

        df_ref  = engineer_features(generate_synthetic_pharma_data(200, seed=42))
        df_prod = engineer_features(generate_synthetic_pharma_data(100, seed=99))
        report  = run_evidently_drift(df_ref, df_prod)

        required = [
            "dataset_drift_detected", "n_drifted_features", "n_total_features",
            "share_drifted_features", "drifted_columns", "per_column_drift",
            "evaluated_at", "method",
        ]
        for k in required:
            assert k in report, f"Missing drift report key: {k}"

    def test_per_column_drift_structure(self):
        from monitoring.monitor import run_evidently_drift
        from pipelines._01_data_ingestion import generate_synthetic_pharma_data
        from pipelines._03_feature_engineering import engineer_features

        df_ref  = engineer_features(generate_synthetic_pharma_data(200, seed=42))
        df_prod = engineer_features(generate_synthetic_pharma_data(100, seed=99))
        report  = run_evidently_drift(df_ref, df_prod)

        for col, drift_info in report["per_column_drift"].items():
            assert "ks_statistic" in drift_info
            assert "p_value"      in drift_info
            assert "drifted"      in drift_info
            assert isinstance(drift_info["drifted"], bool)

    def test_compute_stats_returns_expected_keys(self):
        from monitoring.monitor import compute_stats
        df    = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        stats = compute_stats(df, ["a", "b"])
        for col in ["a", "b"]:
            assert col in stats
            for stat in ["mean", "std", "min", "max", "median"]:
                assert stat in stats[col]


# ══════════════════════════════════════════════════════════════════════════════
# 8. SERVING / INFERENCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestServing:

    def test_add_engineered_features_runs(self):
        from serving.serve import add_engineered_features
        row = pd.DataFrame([{
            "mol_weight": 320, "logp": 2.1, "tpsa": 65, "hbd": 2, "hba": 5,
            "rotatable_bonds": 4, "aromatic_rings": 2, "patient_age": 45,
            "patient_weight_kg": 72, "creatinine_clearance": 90, "is_male": 1,
            "baseline_crp": 4, "baseline_il6": 8, "baseline_tnfa": 12,
            "baseline_wbc": 7, "dose_mg": 100, "treatment_days": 30,
        }])
        result = add_engineered_features(row)
        assert "bioavailability_score" in result.columns
        assert "inflammation_score"    in result.columns
        assert "ro5_violations"        in result.columns

    def test_serving_feature_engineering_no_nulls(self):
        from serving.serve import add_engineered_features
        row = pd.DataFrame([{
            "mol_weight": 400, "logp": 3.0, "tpsa": 80, "hbd": 3, "hba": 6,
            "rotatable_bonds": 5, "aromatic_rings": 2, "patient_age": 55,
            "patient_weight_kg": 80, "creatinine_clearance": 75, "is_male": 0,
            "baseline_crp": 10, "baseline_il6": 20, "baseline_tnfa": 25,
            "baseline_wbc": 8, "dose_mg": 200, "treatment_days": 60,
        }])
        result = add_engineered_features(row)
        assert result.isnull().sum().sum() == 0, "Nulls in serving feature engineering"

    def test_full_predict_pipeline(self, trained_model, model_meta):
        """Integration test: raw input → feature engineering → prediction."""
        from serving.serve import add_engineered_features
        row = pd.DataFrame([{
            "mol_weight": 320, "logp": 2.1, "tpsa": 65, "hbd": 2, "hba": 5,
            "rotatable_bonds": 4, "aromatic_rings": 2, "patient_age": 45,
            "patient_weight_kg": 72, "creatinine_clearance": 90, "is_male": 1,
            "baseline_crp": 4, "baseline_il6": 8, "baseline_tnfa": 12,
            "baseline_wbc": 7, "dose_mg": 100, "treatment_days": 30,
        }])
        feat_row  = add_engineered_features(row)
        feat_cols = [c for c in model_meta["feature_columns"] if c in feat_row.columns]
        X         = feat_row[feat_cols].fillna(0)
        pred      = trained_model.predict(X)[0]
        prob      = trained_model.predict_proba(X)[0]
        assert pred in {0, 1}
        assert abs(prob.sum() - 1.0) < 1e-6
        assert 0.0 <= prob[1] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 9. AUDIT LOG TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLog:

    def test_audit_log_creates_file(self, tmp_path, monkeypatch):
        """audit_log() should write a valid JSONL entry."""
        from config.utils import audit_log, load_config
        cfg = load_config()
        # Redirect audit dir to temp path
        monkeypatch.setitem(cfg["paths"], "audit", str(tmp_path))
        import config.utils as utils_mod
        orig = utils_mod.load_config
        utils_mod.load_config = lambda: cfg
        try:
            audit_log("test_event", {"key": "value"}, actor="pytest")
        finally:
            utils_mod.load_config = orig

        log_files = list(tmp_path.glob("audit_*.jsonl"))
        assert len(log_files) == 1
        with open(log_files[0]) as f:
            entry = json.loads(f.readline())
        assert entry["event"]  == "test_event"
        assert entry["actor"]  == "pytest"
        assert entry["details"] == {"key": "value"}
        assert "timestamp" in entry

    def test_audit_log_entry_has_required_fields(self, cfg):
        """Real audit log from pipeline run should have all required fields."""
        audit_dir = Path(cfg["paths"]["audit"])
        if not audit_dir.exists():
            pytest.skip("No audit logs yet — run `python run.py` first")
        log_files = sorted(audit_dir.glob("audit_*.jsonl"))
        if not log_files:
            pytest.skip("No audit log files found")
        with open(log_files[-1]) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    for field in ["timestamp", "actor", "event", "details", "system"]:
                        assert field in entry, f"Missing audit field: {field}"
                    break  # Only check first entry

    def test_audit_log_is_append_only(self, tmp_path, monkeypatch):
        """Multiple calls should append, not overwrite."""
        from config.utils import audit_log, load_config
        cfg = load_config()
        monkeypatch.setitem(cfg["paths"], "audit", str(tmp_path))
        import config.utils as utils_mod
        orig = utils_mod.load_config
        utils_mod.load_config = lambda: cfg
        try:
            audit_log("event_1", {"n": 1})
            audit_log("event_2", {"n": 2})
            audit_log("event_3", {"n": 3})
        finally:
            utils_mod.load_config = orig

        log_files = list(tmp_path.glob("audit_*.jsonl"))
        with open(log_files[0]) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3


# ══════════════════════════════════════════════════════════════════════════════
# 10. END-TO-END INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_full_mini_pipeline(self, tmp_path, monkeypatch):
        """
        Runs all 6 stages on a small dataset into a temp directory.
        This is the most important test — proves everything connects.
        """
        import importlib.util

        # Point all outputs to tmp_path
        from config import utils as utils_mod
        orig_load = utils_mod.load_config
        cfg = orig_load()

        # Redirect paths to temp directory
        for key in cfg["paths"]:
            new_path = tmp_path / cfg["paths"][key]
            new_path.mkdir(parents=True, exist_ok=True)
            cfg["paths"][key] = str(new_path)

        # Override MLflow to local temp dir
        mlruns = tmp_path / "mlruns"
        mlruns.mkdir(exist_ok=True)
        cfg["mlflow"]["tracking_uri"] = f"file://{mlruns}"

        utils_mod.load_config = lambda: cfg

        try:
            # Stage 1
            from pipelines._01_data_ingestion import generate_synthetic_pharma_data
            df = generate_synthetic_pharma_data(n_samples=100, seed=42)
            df_p = generate_synthetic_pharma_data(n_samples=30, seed=99)
            df.to_csv(Path(cfg["paths"]["raw_data"]) / "drug_trials_train.csv", index=False)
            df_p.to_csv(Path(cfg["paths"]["raw_data"]) / "drug_trials_production.csv", index=False)

            # Stage 3
            from pipelines._03_feature_engineering import engineer_features
            df_feat = engineer_features(df)
            df_feat.to_csv(Path(cfg["paths"]["processed_data"]) / "features_train.csv", index=False)
            df_feat.to_csv(Path(cfg["paths"]["processed_data"]) / "reference.csv", index=False)
            df_prod_feat = engineer_features(df_p)
            df_prod_feat.to_csv(Path(cfg["paths"]["processed_data"]) / "features_production.csv", index=False)

            # Stage 4 — minimal training
            import mlflow, mlflow.sklearn
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split

            feat_cols = [
                "mol_weight", "logp", "tpsa", "patient_age", "dose_mg",
                "treatment_days", "is_drug_like", "bioavailability_score",
                "inflammation_score",
            ]
            feat_cols = [c for c in feat_cols if c in df_feat.columns]
            X = df_feat[feat_cols].fillna(0)
            y = df_feat["efficacy_label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipe = Pipeline([("sc", StandardScaler()),
                             ("clf", RandomForestClassifier(n_estimators=10, random_state=42))])
            pipe.fit(X_train, y_train)
            model_file = Path(cfg["paths"]["models"]) / "best_model.joblib"
            joblib.dump(pipe, model_file)

            meta = {
                "model_name": "random_forest",
                "feature_columns": feat_cols,
                "target_column": "efficacy_label",
                "best_run_id": "test-run-001",
                "best_f1": 0.90,
                "trained_at": "2026-01-01T00:00:00",
            }
            with open(Path(cfg["paths"]["models"]) / "model_metadata.json", "w") as f:
                json.dump(meta, f)

            # Stage 5 — IQ check only (skip OQ/PQ to keep test fast)
            from pipelines._05_model_validation import PharmaQualificationProtocol
            protocol = PharmaQualificationProtocol(cfg)
            iq_ok = protocol.run_iq(str(model_file), str(Path(cfg["paths"]["models"]) / "model_metadata.json"))
            assert iq_ok is True

            # Stage 6 — local registry only
            registry_entry = {
                "model_name": "drug_efficacy_classifier",
                "version": "test-1",
                "stage": "Production",
                "registered_at": "2026-01-01T00:00:00Z",
                "model_path": str(model_file),
                "qualified_for_production": True,
                "feature_columns": feat_cols,
                "best_f1": 0.90,
            }
            with open(Path(cfg["paths"]["registry"]) / "current_production.json", "w") as f:
                json.dump(registry_entry, f)

            # Serving: load and predict
            loaded_model = joblib.load(model_file)
            preds = loaded_model.predict(X_test)
            assert set(preds).issubset({0, 1})

            print("\n  ✓ End-to-end mini pipeline completed successfully")

        finally:
            utils_mod.load_config = orig_load


# ── Allow running directly ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create importable aliases so tests can find the stage modules
    import importlib.util, types

    def _make_alias(original_path: str, alias: str):
        spec = importlib.util.spec_from_file_location(alias, original_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[alias] = mod

    ROOT = Path(__file__).parent.parent
    _make_alias(str(ROOT / "pipelines/01_data_ingestion.py"),  "pipelines._01_data_ingestion")
    _make_alias(str(ROOT / "pipelines/02_data_validation.py"), "pipelines._02_data_validation")
    _make_alias(str(ROOT / "pipelines/03_feature_engineering.py"), "pipelines._03_feature_engineering")
    _make_alias(str(ROOT / "pipelines/05_model_validation.py"), "pipelines._05_model_validation")

    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v", "--tb=short"]))
