# ui/dashboard.py
# Streamlit dashboard - unified view of all pipeline systems
# Run: streamlit run ui/dashboard.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from config.utils import load_config

cfg = load_config()

st.set_page_config(
    page_title="Pharma MLOps Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

SERVE_URL   = f"http://localhost:{cfg['serving']['port']}"
MONITOR_URL = f"http://localhost:{cfg['monitoring']['port']}"


def safe_get(url, timeout=3):
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def safe_post(url, data, timeout=5):
    try:
        r = requests.post(url, json=data, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60/1a1a2e/ffffff?text=Pharma+MLOps", width=200)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    " Overview",
    " Predict Efficacy",
    " Model Performance",
    " Data Drift",
    " Audit Log",
    " Pipeline Status",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")

# Quick health checks
serve_health   = safe_get(f"{SERVE_URL}/health")
monitor_health = safe_get(f"{MONITOR_URL}/health")

col1, col2 = st.sidebar.columns(2)
col1.metric("Serving",    " OK" if serve_health   else " Off")
col2.metric("Monitoring", " OK" if monitor_health else " Off")

mlflow_ok = False
try:
    r = requests.get("http://127.0.0.1:5000/health", timeout=2)
    mlflow_ok = r.status_code == 200
except Exception:
    pass
st.sidebar.metric("MLflow", " OK" if mlflow_ok else " Off")

# ── Pages ──────────────────────────────────────────────────────────────────────

if page == " Overview":
    st.title(" Pharma MLOps Dashboard")
    st.markdown("**Local end-to-end ML pipeline for drug efficacy prediction**")

    col1, col2, col3, col4 = st.columns(4)

    # Model info
    model_info = safe_get(f"{SERVE_URL}/model/info")
    if model_info:
        col1.metric("Model Type",    model_info.get("model_type", "N/A"))
        col2.metric("Model Version", f"v{model_info.get('version', 'N/A')}")
        col3.metric("Best F1 Score", f"{model_info.get('best_f1', 0):.4f}")
        col4.metric("Status",        model_info.get("stage", "N/A"))
    else:
        col1.metric("Serving API", "Offline")

    st.markdown("---")

    # Pipeline stages status
    st.subheader("Pipeline Stage Status")
    proc_path = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["models"])
    registry_path = Path(cfg["paths"]["registry"])

    stages = {
        "1. Data Ingestion":     (Path(cfg["paths"]["raw_data"]) / "drug_trials_train.csv").exists(),
        "2. Data Validation":    (proc_path / "validation_report.json").exists(),
        "3. Feature Engineering":(proc_path / "features_train.csv").exists(),
        "4. Model Training":     (model_path / "best_model.joblib").exists(),
        "5. IQ/OQ/PQ Validation":(proc_path / "qualification_report.json").exists(),
        "6. Model Registry":     (registry_path / "current_production.json").exists(),
        "7. Serving API":        serve_health is not None,
        "8. Monitoring":         monitor_health is not None,
    }

    stage_cols = st.columns(4)
    for i, (stage, done) in enumerate(stages.items()):
        stage_cols[i % 4].metric(stage, " Done" if done else "⏳ Pending")

    # Recent audit events
    st.markdown("---")
    st.subheader("Recent Audit Events")
    audit_dir = Path(cfg["paths"]["audit"])
    if audit_dir.exists():
        logs = []
        for f in sorted(audit_dir.glob("*.jsonl"), reverse=True)[:3]:
            with open(f) as fh:
                for line in fh:
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        pass
        if logs:
            df_logs = pd.DataFrame(logs[-10:][::-1])[["timestamp", "event", "actor"]]
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No audit events yet.")
    else:
        st.info("Audit directory not found.")


elif page == " Predict Efficacy":
    st.title(" Drug Efficacy Prediction")
    st.markdown("Enter drug compound and patient parameters to get an efficacy prediction.")

    if not serve_health:
        st.error("⚠ Serving API is offline. Run `python serving/serve.py` first.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Molecular Properties")
            mol_weight = st.slider("Molecular Weight (Da)", 100.0, 700.0, 350.0)
            logp       = st.slider("LogP",  -2.0, 7.0, 2.5)
            tpsa       = st.slider("TPSA",   20.0, 200.0, 75.0)
            hbd        = st.slider("H-Bond Donors (HBD)", 0, 10, 2)
            hba        = st.slider("H-Bond Acceptors (HBA)", 0, 12, 5)
            rot_bonds  = st.slider("Rotatable Bonds", 0, 15, 4)
            arom_rings = st.slider("Aromatic Rings", 0, 5, 2)

        with col2:
            st.subheader("Patient Profile")
            age        = st.slider("Age (years)", 18, 90, 45)
            weight     = st.slider("Weight (kg)", 40.0, 150.0, 70.0)
            crcl       = st.slider("Creatinine Clearance (mL/min)", 15.0, 150.0, 90.0)
            is_male    = st.selectbox("Sex", ["Male", "Female"])

        with col3:
            st.subheader("Clinical Parameters")
            crp        = st.slider("Baseline CRP (mg/L)",  0.0, 80.0, 5.0)
            il6        = st.slider("Baseline IL-6 (pg/mL)", 0.0, 100.0, 10.0)
            tnfa       = st.slider("Baseline TNF-α", 0.0, 100.0, 15.0)
            wbc        = st.slider("WBC (10⁹/L)", 2.0, 20.0, 7.0)
            dose       = st.selectbox("Dose (mg)", [25, 50, 100, 200, 400])
            days       = st.slider("Treatment Days", 7, 90, 30)

        if st.button("🔬 Predict Efficacy", type="primary"):
            payload = {
                "mol_weight": mol_weight, "logp": logp, "tpsa": tpsa,
                "hbd": float(hbd), "hba": float(hba),
                "rotatable_bonds": float(rot_bonds), "aromatic_rings": float(arom_rings),
                "patient_age": float(age), "patient_weight_kg": weight,
                "creatinine_clearance": crcl,
                "is_male": 1.0 if is_male == "Male" else 0.0,
                "baseline_crp": crp, "baseline_il6": il6,
                "baseline_tnfa": tnfa, "baseline_wbc": wbc,
                "dose_mg": float(dose), "treatment_days": float(days),
            }

            with st.spinner("Running inference..."):
                result = safe_post(f"{SERVE_URL}/predict", payload)

            if result:
                st.markdown("---")
                rcol1, rcol2, rcol3 = st.columns(3)

                pred = result["prediction"]
                p_eff = result["probability_effective"]
                conf  = result["confidence"]

                rcol1.metric(
                    "Prediction",
                    " EFFECTIVE" if pred == 1 else " INEFFECTIVE",
                )
                rcol2.metric("Efficacy Probability", f"{p_eff:.1%}")
                rcol3.metric("Confidence", conf.upper())

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_eff * 100,
                    title={"text": "Efficacy Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "green" if pred == 1 else "red"},
                        "steps": [
                            {"range": [0, 40],  "color": "#ffcccc"},
                            {"range": [40, 60], "color": "#fff3cc"},
                            {"range": [60, 100],"color": "#ccffcc"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Lipinski check
                violations = sum([
                    mol_weight > 500, logp > 5, hbd > 5, hba > 10
                ])
                st.info(f"Lipinski Rule of Five: {violations} violation(s) — {'Drug-like ✓' if violations <= 1 else 'Poor bioavailability risk ⚠'}")
            else:
                st.error("Prediction failed. Check serving API.")


elif page == " Model Performance":
    st.title(" Model Performance")

    proc_path  = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["models"])

    # Qualification report
    qual_file = proc_path / "qualification_report.json"
    if qual_file.exists():
        with open(qual_file) as f:
            qual = json.load(f)

        st.subheader("IQ/OQ/PQ Qualification Status")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("IQ", " PASSED" if qual["iq_result"] == "PASSED" else " FAILED")
        q2.metric("OQ", " PASSED" if qual["oq_result"] == "PASSED" else " FAILED")
        q3.metric("PQ", " PASSED" if qual["pq_result"] == "PASSED" else " FAILED")
        q4.metric("Overall", qual["overall_qualification_status"])

        # OQ metrics
        if "oq" in qual.get("detailed_results", {}):
            st.subheader("Model Metrics (OQ)")
            oq  = qual["detailed_results"]["oq"]
            metrics = oq.get("metrics", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
            m2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            m3.metric("Recall",    f"{metrics.get('recall', 0):.4f}")
            m4.metric("F1 Score",  f"{metrics.get('f1_score', 0):.4f}")
            m5.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.4f}")

            # Confusion matrix
            cm = oq.get("confusion_matrix", {})
            if cm:
                st.subheader("Confusion Matrix")
                cm_data = pd.DataFrame(
                    [[cm.get("TP", 0), cm.get("FP", 0)],
                     [cm.get("FN", 0), cm.get("TN", 0)]],
                    columns=["Predicted Positive", "Predicted Negative"],
                    index=["Actual Positive", "Actual Negative"]
                )
                fig = px.imshow(
                    cm_data.values,
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    x=["Positive", "Negative"], y=["Positive", "Negative"],
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline first to see performance metrics.")

    # Feature importance
    fi_files = list(model_path.glob("feature_importance_*.csv")) if model_path.exists() else []
    if fi_files:
        st.subheader("Feature Importance (Best Model)")
        df_fi = pd.read_csv(fi_files[0]).head(15)
        fig = px.bar(df_fi, x="importance", y="feature", orientation="h",
                     title="Top 15 Features", color="importance",
                     color_continuous_scale="teal")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)


elif page == " Data Drift":
    st.title(" Data Drift Monitoring")

    if not monitor_health:
        st.error("⚠ Monitoring API is offline. Run `python monitoring/monitor.py` first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("Compares **reference (training)** vs **production** distributions.")
        with col2:
            if st.button(" Run Drift Check Now"):
                with st.spinner("Running Evidently drift analysis..."):
                    result = safe_post(f"{MONITOR_URL}/drift/run", {})
                if result:
                    st.success("Drift check complete!")
                else:
                    st.error("Drift check failed.")

        # Latest drift report
        drift = safe_get(f"{MONITOR_URL}/drift/latest")
        if drift and "message" not in drift:
            st.markdown("---")
            d1, d2, d3 = st.columns(3)
            d1.metric("Drift Detected", "⚠ YES" if drift.get("dataset_drift_detected") else " NO")
            d2.metric("Drifted Features",
                      f"{drift.get('n_drifted_features', 0)} / {drift.get('n_total_features', '?')}")
            d3.metric("Drift Share", f"{drift.get('share_drifted_features', 0):.1%}")

            # Per-column drift table
            col_drift = drift.get("per_column_drift", {})
            if col_drift:
                st.subheader("Per-Feature Drift (KS Test)")
                df_drift = pd.DataFrame([
                    {"Feature": k, "KS Statistic": v.get("ks_statistic"), "P-Value": v.get("p_value"),
                     "Drifted": "⚠ YES" if v.get("drifted") else " NO"}
                    for k, v in col_drift.items()
                ])
                st.dataframe(df_drift.sort_values("KS Statistic", ascending=False), use_container_width=True)

            # Distribution comparison chart
            ref_stats  = drift.get("ref_stats", {})
            prod_stats = drift.get("prod_stats", {})
            if ref_stats and prod_stats:
                st.subheader("Mean Comparison: Reference vs Production")
                features = [k for k in ref_stats if k in prod_stats]
                ref_means  = [ref_stats[k]["mean"]  for k in features]
                prod_means = [prod_stats[k]["mean"] for k in features]
                fig = go.Figure(data=[
                    go.Bar(name="Reference",  x=features, y=ref_means,  marker_color="#4e79a7"),
                    go.Bar(name="Production", x=features, y=prod_means, marker_color="#f28e2b"),
                ])
                fig.update_layout(barmode="group", title="Feature Means: Reference vs Production",
                                  xaxis_tickangle=-45, height=450)
                st.plotly_chart(fig, use_container_width=True)
        elif drift and "message" in drift:
            st.info(drift["message"])
        else:
            st.info("No drift data yet. Click 'Run Drift Check Now'.")


elif page == " Audit Log":
    st.title(" GxP Audit Log")
    st.markdown("Immutable, timestamped audit trail of all pipeline events (21 CFR Part 11 compliant format).")

    audit_dir = Path(cfg["paths"]["audit"])
    if not audit_dir.exists():
        st.info("No audit logs found. Run the pipeline first.")
    else:
        logs = []
        for f in sorted(audit_dir.glob("*.jsonl"), reverse=True):
            with open(f) as fh:
                for line in fh:
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        pass

        if logs:
            st.metric("Total Events", len(logs))
            df_logs = pd.DataFrame(logs[::-1])  # newest first
            cols_show = ["timestamp", "event", "actor", "system", "version"]
            available = [c for c in cols_show if c in df_logs.columns]
            st.dataframe(df_logs[available], use_container_width=True)

            # Event type distribution
            if "event" in df_logs.columns:
                st.subheader("Event Distribution")
                counts = df_logs["event"].value_counts().reset_index()
                counts.columns = ["Event", "Count"]
                fig = px.bar(counts, x="Event", y="Count", color="Count", color_continuous_scale="teal")
                st.plotly_chart(fig, use_container_width=True)

            # Expandable raw entries
            st.subheader("Raw Log Entries")
            for entry in logs[-5:][::-1]:
                with st.expander(f"{entry.get('timestamp', '')} — {entry.get('event', '')}"):
                    st.json(entry)
        else:
            st.info("No audit events logged yet.")


elif page == " Pipeline Status":
    st.title(" Pipeline Status")

    proc_path     = Path(cfg["paths"]["processed_data"])
    model_path    = Path(cfg["paths"]["models"])
    registry_path = Path(cfg["paths"]["registry"])
    raw_path      = Path(cfg["paths"]["raw_data"])

    st.subheader("Data Files")
    files = {
        "Raw training data":      raw_path / "drug_trials_train.csv",
        "Raw production data":    raw_path / "drug_trials_production.csv",
        "Validation report":      proc_path / "validation_report.json",
        "Engineered features":    proc_path / "features_train.csv",
        "Reference dataset":      proc_path / "reference.csv",
        "Qualification report":   proc_path / "qualification_report.json",
        "Best model":             model_path / "best_model.joblib",
        "Model metadata":         model_path / "model_metadata.json",
        "Registry (production)":  registry_path / "current_production.json",
    }

    for label, path in files.items():
        exists = path.exists()
        size   = f"{path.stat().st_size / 1024:.1f} KB" if exists else "—"
        mtime  = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else "—"
        c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
        c1.markdown(f"**{label}**")
        c2.markdown("" if exists else "")
        c3.markdown(size)
        c4.markdown(mtime)

    # Feature store info
    st.markdown("---")
    st.subheader("Feature Store")
    feat_db = Path(cfg["paths"]["features"]) / "feature_store.db"
    if feat_db.exists():
        import sqlite3
        with sqlite3.connect(str(feat_db)) as conn:
            try:
                rows = pd.read_sql("SELECT * FROM feature_sets ORDER BY created_at DESC", conn)
                st.dataframe(rows[["feature_set_name", "version", "created_at", "n_samples", "n_features"]])
            except Exception as e:
                st.warning(f"Feature store read error: {e}")
    else:
        st.info("Feature store not initialized yet.")

    # MLflow link
    st.markdown("---")
    st.subheader("MLflow Experiment Tracking")
    st.markdown(f"[Open MLflow UI →](http://localhost:5000) *(must be running)*")

    # Model registry
    if (registry_path / "current_production.json").exists():
        st.subheader("Current Production Model")
        with open(registry_path / "current_production.json") as f:
            reg = json.load(f)
        st.json(reg)
