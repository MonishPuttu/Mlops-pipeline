# /brag plan — Mlops-pipeline

**What it is:** A local, end-to-end MLOps pipeline for drug-efficacy prediction — ingestion → validation → feature engineering → training → GxP-style validation → registry → serving → monitoring, with automatic retraining on data drift.
**Who it's for:** ML/MLOps engineers and hiring managers who want to see production ML practice, not just a notebook.
**What sets it apart:** It closes the loop: a KS-test drift monitor detects shift in production data and re-runs the whole pipeline, registering a new qualified model version.
**Most impressive claim:** Drift detected (25% of features > 15% threshold) → retrained, re-qualified and promoted v2 in 24s.
**Visual hook:** A red "⚠ DRIFT ALERT" with the real monitoring log lines.
**Tone:** cinematic-tech — dark terminal, driving A-minor pulse.
**Share caption:** "Train. Validate. Ship. Watch. Retrain."

## Every number is from a real run
The pipeline was run in a scratch copy of this repo (stages 01–06 run directly, MLflow on a local file store, MinIO unreachable so local files were used):
- Stage 4: logistic_regression F1 0.9728 · random_forest F1 0.9863 (ROC-AUC 0.9949) · gradient_boosting F1 0.9810 → best: random_forest
- Stage 5: IQ / OQ / PQ all PASSED — accuracy 0.9750, precision 0.9945, 2.8 ms per 100 samples → OVERALL STATUS: QUALIFIED
- Stage 6: `drug_efficacy_classifier` v1 registered
- `pipelines/retrain_trigger.py` `run_once()`: 3 / 12 features drifted (patient_age, baseline_crp, inflammation_score), share 25% > 15% → full retrain → v2 PRODUCTION-ready, "Retraining complete in 24s"
- patient_age histogram drawn from the real `reference.csv` vs `features_production.csv` (means 48.0 vs 53.7)

## Visual identity
- Reference/production colors `#4e79a7` / `#f28e2b` from the drift chart in `ui/dashboard.py`
- Architecture flow from the README diagram; stack pills from the docker-compose files, `dags/`, `dvc.yaml` and `kubernetes/`
- Space Grotesk + JetBrains Mono (the project has no web font of its own)

## Storyboard (21s, 1920×1080 @ 30fps)
| # | Time | Scene | On screen |
|---|------|-------|-----------|
| 1 | 0.0–3.4 | **Hook** | ⚠ DRIFT ALERT — "Your model's data just changed." + real monitor log |
| 2 | 3.4–7.2 | **Reveal** | "Pharma MLOps Pipeline" + the 7-stage flow lighting up, "↺ retrain on drift" loop |
| 3 | 7.2–10.8 | **Training** | Three-model F1 race (real scores), random_forest wins |
| 4 | 10.8–14.2 | **Validation** | IQ / OQ / PQ cards pass → "OVERALL STATUS: QUALIFIED" stamp |
| 5 | 14.2–18.0 | **Self-healing** | patient_age drift histogram, drift share 25% crosses 15% line, 6 stages re-run, v1 → v2 PRODUCTION |
| 6 | 18.0–21.0 | **Outro** | "Train. Validate. Ship. Watch. Retrain." + stack + GitHub link |
