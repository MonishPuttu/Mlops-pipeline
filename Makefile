# Makefile — pharma-mlops (macOS Apple Silicon)
# Place at: ~/pharma-mlops/Makefile

.PHONY: help prereqs k3d infra-up infra-down infra-logs infra-ps verify-day1 \
        pipeline test serve monitor dashboard clean

help:
	@echo ""
	@echo "pharma-mlops — Available commands"
	@echo "────────────────────────────────────────────────────"
	@echo ""
	@echo "  SETUP"
	@echo "  make prereqs      Check all prerequisites (run first)"
	@echo "  make k3d          Install k3d + create k3s cluster"
	@echo ""
	@echo "  INFRASTRUCTURE  (Day 1)"
	@echo "  make infra-up     Start MinIO, PostgreSQL, Redis, pgAdmin"
	@echo "  make infra-down   Stop infrastructure containers"
	@echo "  make infra-logs   Tail infrastructure logs"
	@echo "  make infra-ps     Show container status"
	@echo "  make verify-day1  Run all Day 1 verification checks"
	@echo ""
	@echo "  PIPELINE  (existing)"
	@echo "  make pipeline     Run the full ML pipeline (python run.py)"
	@echo "  make test         Run pytest test suite"
	@echo "  make serve        Start model serving API"
	@echo "  make monitor      Start drift monitoring service"
	@echo "  make dashboard    Start Streamlit dashboard"
	@echo ""
	@echo "  CLEANUP"
	@echo "  make clean        Stop + delete all infra volumes (DESTRUCTIVE)"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────
prereqs:
	@bash infra/check_prereqs.sh

k3d:
	@bash infra/install_k3d.sh

# ── Infrastructure ─────────────────────────────────────────────
infra-up:
	@echo "Starting infrastructure services..."
	docker compose -f docker-compose.infra.yml up -d
	@echo "Waiting 20s for services to initialize..."
	@sleep 20
	@docker compose -f docker-compose.infra.yml ps

infra-down:
	docker compose -f docker-compose.infra.yml down

infra-logs:
	docker compose -f docker-compose.infra.yml logs -f

infra-ps:
	docker compose -f docker-compose.infra.yml ps

verify-day1:
	@bash infra/verify_day1.sh

# ── Pipeline (your existing commands, unchanged) ───────────────
pipeline:
	python run.py

test:
	python -m pytest tests/ -v

serve:
	python serving/serve.py

monitor:
	python monitoring/monitor.py

dashboard:
	streamlit run ui/dashboard.py

# ── Cleanup ────────────────────────────────────────────────────
clean:
	@echo "WARNING: Deletes all containers and data volumes."
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ]
	docker compose -f docker-compose.infra.yml down -v
	@echo "Done."

mlflow-up:
	docker compose -f docker-compose.mlflow.yml up -d
	@echo "Waiting 25s for MLflow to initialize..."
	@sleep 25
	@docker compose -f docker-compose.mlflow.yml ps

mlflow-down:
	docker compose -f docker-compose.mlflow.yml down

mlflow-logs:
	docker compose -f docker-compose.mlflow.yml logs -f

verify-day2:
	@bash infra/verify_day2.sh

verify-day3:
	@bash infra/verify_day3.sh

airflow-init:
	docker compose -f docker-compose.airflow.yml run --rm airflow-init

airflow-up:
	docker compose -f docker-compose.airflow.yml up -d airflow-webserver airflow-scheduler
	@echo "Waiting 40s for Airflow to initialize..."
	@sleep 40
	@docker compose -f docker-compose.airflow.yml ps

airflow-down:
	docker compose -f docker-compose.airflow.yml down

airflow-logs:
	docker compose -f docker-compose.airflow.yml logs -f

verify-day4:
	@bash infra/verify_day4.sh