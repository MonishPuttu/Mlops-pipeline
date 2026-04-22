from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "pharma-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

def check_drift(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from monitoring.monitor import load_dataframes, run_evidently_drift
    from config.utils import load_config, audit_log

    cfg = load_config()
    df_ref, df_prod = load_dataframes()

    if df_ref is None or df_prod is None:
        print("Data not available — skipping drift check")
        return "no_drift"

    report = run_evidently_drift(df_ref, df_prod)
    drift_share = report.get("share_drifted_features", 0)
    detected = report.get("dataset_drift_detected", False)
    threshold = cfg["validation"]["max_drift_score"]

    context["task_instance"].xcom_push(key="drift_share", value=drift_share)
    context["task_instance"].xcom_push(key="drift_detected", value=detected)
    context["task_instance"].xcom_push(key="drifted_columns", value=report.get("drifted_columns", []))

    print(f"Drift detected: {detected} | Share: {drift_share:.1%} | Threshold: {threshold:.0%}")

    audit_log("drift_check_completed", {
        "drift_detected": detected,
        "drift_share": round(drift_share, 4),
        "drifted_columns": report.get("drifted_columns", []),
    })

    if detected and drift_share > threshold:
        return "trigger_retrain"
    return "no_drift"

with DAG(
    dag_id="pharma_drift_detection",
    default_args=default_args,
    description="Checks for data drift every 4 hours and triggers retraining if needed",
    schedule="0 */4 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["pharma", "monitoring"],
) as dag:

    drift_check = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="pharma_pipeline",
        wait_for_completion=False,
    )

    no_drift = EmptyOperator(task_id="no_drift")

    drift_check >> [trigger_retrain, no_drift]