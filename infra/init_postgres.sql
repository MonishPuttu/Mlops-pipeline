-- infra/init_postgres.sql
-- Runs automatically when the postgres container starts for the first time.

CREATE USER mlflow WITH PASSWORD 'mlflowpass123';
CREATE DATABASE mlflow_db OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;

CREATE USER airflow WITH PASSWORD 'airflowpass123';
CREATE DATABASE airflow_db OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;

CREATE USER pharmaapp WITH PASSWORD 'pharmaapppass123';
CREATE DATABASE pharma_app_db OWNER pharmaapp;
GRANT ALL PRIVILEGES ON DATABASE pharma_app_db TO pharmaapp;

\echo 'Databases created: mlflow_db, airflow_db, pharma_app_db'
