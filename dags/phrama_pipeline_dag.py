from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.insert(0, "/opt/airflow/project")

default_args = {
    "owner": "pharma-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

def run_stage(module_path: str, fn_name: str = "run"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)()

with DAG(
    dag_id="pharma_pipeline",
    default_args=default_args,
    description="Full pharma MLOps pipeline: ingest → validate → features → train → validate → register",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["pharma", "pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_stage,
        op_args=["/opt/airflow/project/pipelines/01_data_ingestion.py"],
    )

    t2 = PythonOperator(
        task_id="data_validation",
        python_callable=run_stage,
        op_args=["/opt/airflow/project/pipelines/02_data_validation.py"],
    )

    t3 = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_stage,
        op_args=["/opt/airflow/project/pipelines/03_feature_engineering.py"],
    )

    t4 = PythonOperator(
        task_id="model_training",
        python_callable=run_stage,
        op_args=["/opt/airflow/project/pipelines/04_model_training.py"],
    )

    t5 = PythonOperator(
        task_id="model_validation",
        python_callable=run_stage,
        op_args=["/opt/airflow/project/pipelines/05_model_validation.py"],
    )

    t6 = PythonOperator(
        task_id="model_registry",
        python_callable=run_stage,
        op_args=["/opt/airflow/project/pipelines/06_model_registry.py"],
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6