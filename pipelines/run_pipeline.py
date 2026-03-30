# pipelines/run_pipeline.py
# Master orchestrator: runs all pipeline stages in sequence
# Usage: python pipelines/run_pipeline.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from config.utils import load_config, get_logger, audit_log, ensure_dirs

console = Console()
logger  = get_logger("pipeline_orchestrator")


def run_stage(name: str, fn, results: list):
    console.rule(f"[bold cyan]{name}[/bold cyan]")
    start = time.perf_counter()
    try:
        output = fn()
        elapsed = time.perf_counter() - start
        results.append({"stage": name, "status": "SUCCESS", "duration_s": round(elapsed, 2), "error": None})
        console.print(f"[green]✓ {name} completed in {elapsed:.2f}s[/green]")
        return output
    except Exception as e:
        elapsed = time.perf_counter() - start
        results.append({"stage": name, "status": "FAILED", "duration_s": round(elapsed, 2), "error": str(e)})
        console.print(f"[red]✗ {name} FAILED: {e}[/red]")
        raise


def main():
    cfg = load_config()
    ensure_dirs()

    console.print(Panel.fit(
        "[bold white]Pharma MLOps Pipeline[/bold white]\n"
        "[dim]Local end-to-end ML pipeline for drug efficacy prediction[/dim]\n"
        f"[dim]Started: {datetime.utcnow().isoformat()}Z[/dim]",
        border_style="cyan"
    ))

    results = []
    pipeline_start = time.perf_counter()

    audit_log("pipeline_started", {"config": cfg["project"]})

    try:
        # Stage 1: Data Ingestion
        from pipelines.data_ingestion_mod import run as ingest
        run_stage("Stage 1: Data Ingestion", ingest, results)

        # Stage 2: Data Validation
        from pipelines.data_validation_mod import run as validate
        run_stage("Stage 2: Data Validation", validate, results)

        # Stage 3: Feature Engineering
        from pipelines.feature_engineering_mod import run as engineer
        run_stage("Stage 3: Feature Engineering", engineer, results)

        # Stage 4: Model Training
        from pipelines.model_training_mod import run as train
        run_stage("Stage 4: Model Training", train, results)

        # Stage 5: Model Validation (IQ/OQ/PQ)
        from pipelines.model_validation_mod import run as qual
        run_stage("Stage 5: IQ/OQ/PQ Validation", qual, results)

        # Stage 6: Model Registry
        from pipelines.model_registry_mod import run as register
        run_stage("Stage 6: Model Registry", register, results)

    except Exception as e:
        console.print(f"\n[red bold]Pipeline halted at failed stage.[/red bold]")
        audit_log("pipeline_failed", {"error": str(e), "completed_stages": [r["stage"] for r in results if r["status"] == "SUCCESS"]})
    finally:
        total_time = time.perf_counter() - pipeline_start

        # Print summary table
        table = Table(title="\nPipeline Run Summary", show_header=True, header_style="bold cyan")
        table.add_column("Stage",    style="white",  min_width=35)
        table.add_column("Status",   style="bold",   min_width=10)
        table.add_column("Duration", style="yellow", min_width=10)
        table.add_column("Error",    style="red",    min_width=20)

        all_passed = True
        for r in results:
            status_str = "[green]SUCCESS[/green]" if r["status"] == "SUCCESS" else "[red]FAILED[/red]"
            err_str    = r["error"][:60] if r["error"] else "—"
            table.add_row(r["stage"], status_str, f"{r['duration_s']}s", err_str)
            if r["status"] != "SUCCESS":
                all_passed = False

        console.print(table)
        console.print(f"\n[bold]Total pipeline time: {total_time:.2f}s[/bold]")

        if all_passed:
            console.print(Panel(
                "[bold green]✓ All stages completed successfully![/bold green]\n\n"
                "Next steps:\n"
                "  [cyan]mlflow server --host 127.0.0.1 --port 5000[/cyan]  (if not running)\n"
                "  [cyan]python serving/serve.py[/cyan]        → Model API on :8000\n"
                "  [cyan]python monitoring/monitor.py[/cyan]   → Drift monitor on :8001\n"
                "  [cyan]streamlit run ui/dashboard.py[/cyan]  → Dashboard on :8501",
                border_style="green"
            ))
            audit_log("pipeline_completed", {
                "total_time_s": round(total_time, 2),
                "stages": results,
                "all_passed": True
            })
        else:
            console.print("[red]Pipeline finished with failures. See logs for details.[/red]")
            audit_log("pipeline_completed", {
                "total_time_s": round(total_time, 2),
                "stages": results,
                "all_passed": False
            })


if __name__ == "__main__":
    main()
