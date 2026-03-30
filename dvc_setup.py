# dvc_setup.py
# Sets up DVC (Data Version Control) for local data versioning.
# DVC tracks your data files the same way Git tracks code.
# Run: python dvc_setup.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess
import json
from pathlib import Path
from config.utils import get_logger, load_config

logger = get_logger("dvc_setup")
cfg    = load_config()


def run_cmd(cmd: list, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    logger.info(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=capture, text=True, check=False
    )
    if result.returncode != 0 and check:
        logger.error(f"Command failed: {result.stderr}")
    elif result.stdout.strip():
        logger.info(f"    → {result.stdout.strip()}")
    return result


def setup_dvc():
    logger.info("=" * 60)
    logger.info("DVC DATA VERSION CONTROL SETUP")
    logger.info("=" * 60)

    # ── 1. Check Git repo ─────────────────────────────────────────
    logger.info("\n[1/6] Checking Git repository...")
    git_check = run_cmd(["git", "rev-parse", "--git-dir"], check=False)
    if git_check.returncode != 0:
        logger.info("  Initializing Git repo...")
        run_cmd(["git", "init"])
        run_cmd(["git", "config", "user.email", "pharma-mlops@local.dev"])
        run_cmd(["git", "config", "user.name",  "Pharma MLOps"])
    else:
        logger.info("  Git repo already exists ✓")

    # ── 2. Init DVC ───────────────────────────────────────────────
    logger.info("\n[2/6] Initializing DVC...")
    dvc_dir = Path(".dvc")
    if dvc_dir.exists():
        logger.info("  DVC already initialized ✓")
    else:
        r = run_cmd(["dvc", "init"], check=False)
        if r.returncode != 0:
            logger.error("DVC not installed. Install with: pip install dvc")
            logger.info("Skipping DVC setup — creating .dvc config manually instead.")
            _create_manual_dvc_config()
            return
        logger.info("  DVC initialized ✓")

    # ── 3. Configure local remote ─────────────────────────────────
    logger.info("\n[3/6] Configuring local DVC remote storage...")
    remote_path = Path("dvc_storage").resolve()
    remote_path.mkdir(exist_ok=True)
    run_cmd(["dvc", "remote", "add", "--default", "local_storage", str(remote_path)], check=False)
    run_cmd(["dvc", "remote", "modify", "local_storage", "type", "local"], check=False)
    logger.info(f"  Remote: {remote_path} ✓")

    # ── 4. Track data files ───────────────────────────────────────
    logger.info("\n[4/6] Tracking data files with DVC...")

    files_to_track = [
        Path(cfg["paths"]["raw_data"])       / "drug_trials_train.csv",
        Path(cfg["paths"]["raw_data"])       / "drug_trials_production.csv",
        Path(cfg["paths"]["processed_data"]) / "features_train.csv",
        Path(cfg["paths"]["processed_data"]) / "reference.csv",
        Path(cfg["paths"]["processed_data"]) / "features_production.csv",
        Path(cfg["paths"]["models"])         / "best_model.joblib",
    ]

    tracked = []
    for f in files_to_track:
        if f.exists():
            r = run_cmd(["dvc", "add", str(f)], check=False)
            if r.returncode == 0:
                tracked.append(str(f))
                logger.info(f"  Tracked: {f.name} ✓")
            else:
                logger.warning(f"  Could not track {f.name}: {r.stderr.strip()[:80]}")
        else:
            logger.warning(f"  File not found (run pipeline first): {f}")

    # ── 5. Git commit the .dvc files ──────────────────────────────
    logger.info("\n[5/6] Committing DVC metadata to Git...")
    dvc_files = list(Path(".").glob("**/*.dvc")) + [Path(".dvc")]

    # Create .gitignore for data directories
    gitignore_path = Path(".gitignore")
    gitignore_lines = set()
    if gitignore_path.exists():
        gitignore_lines = set(gitignore_path.read_text().splitlines())

    new_lines = [
        "# Python",
        "__pycache__/", "*.pyc", "*.pyo", ".pytest_cache/",
        "# Data (tracked by DVC)",
        "data/raw/*.csv", "data/processed/*.csv",
        "models/trained/*.joblib",
        "# Logs & runtime",
        "logs/", "audit/", "mlruns/", "dvc_storage/",
        "venv/", ".env",
    ]
    for line in new_lines:
        gitignore_lines.add(line)

    gitignore_path.write_text("\n".join(sorted(gitignore_lines)) + "\n")
    logger.info("  .gitignore updated ✓")

    run_cmd(["git", "add", ".gitignore", ".dvc/", "*.dvc",
             "data/", "models/", "config/", "pipelines/",
             "monitoring/", "serving/", "ui/", "tests/",
             "requirements.txt", "run.py", "setup.sh", "README.md"], check=False)

    run_cmd(["git", "commit", "-m",
             f"chore: initialize DVC tracking for pharma-mlops pipeline\n\n"
             f"Tracked {len(tracked)} data files."], check=False)

    # ── 6. Push to local DVC remote ───────────────────────────────
    logger.info("\n[6/6] Pushing data to local DVC remote...")
    r = run_cmd(["dvc", "push"], check=False)
    if r.returncode == 0:
        logger.info("  Data pushed to local remote ✓")
    else:
        logger.warning(f"  Push failed (may need to re-add files): {r.stderr.strip()[:120]}")

    # ── Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DVC SETUP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Tracked files : {len(tracked)}")
    logger.info(f"  Remote storage: {remote_path}")
    logger.info("""
  Common DVC commands:
    dvc status          — check if tracked files have changed
    dvc diff            — show changes vs last commit
    dvc push            — push data to remote storage
    dvc pull            — pull data from remote storage
    dvc repro           — reproduce the pipeline if inputs changed

  To version a new dataset:
    1. Replace/update data files
    2. dvc add data/raw/drug_trials_train.csv
    3. git add data/raw/drug_trials_train.csv.dvc
    4. git commit -m "data: update training dataset v2"
    5. dvc push
""")


def _create_manual_dvc_config():
    """
    Creates DVC-compatible metadata files manually when DVC isn't installed.
    Generates .dvc tracking files and a dvc.yaml pipeline definition.
    """
    import hashlib
    logger.info("\nCreating manual DVC-compatible tracking files...")

    dvc_dir = Path(".dvc")
    dvc_dir.mkdir(exist_ok=True)
    (dvc_dir / "config").write_text(
        "[core]\n    autostage = true\n[remote \"local_storage\"]\n    url = dvc_storage\n"
    )

    files_to_track = [
        Path(cfg["paths"]["raw_data"])       / "drug_trials_train.csv",
        Path(cfg["paths"]["processed_data"]) / "features_train.csv",
        Path(cfg["paths"]["models"])         / "best_model.joblib",
    ]

    for f in files_to_track:
        if not f.exists():
            continue
        content = f.read_bytes()
        md5 = hashlib.md5(content).hexdigest()
        dvc_meta = {
            "outs": [{
                "md5": md5,
                "size": len(content),
                "hash": "md5",
                "path": str(f),
            }]
        }
        dvc_file = Path(f"{f}.dvc")
        dvc_file.write_text(
            f"# DVC metadata for {f.name}\n"
            + json.dumps(dvc_meta, indent=2)
        )
        logger.info(f"  Created: {dvc_file.name} (md5={md5[:8]}...)")

    # Write pipeline definition
    dvc_yaml = """# dvc.yaml - Pipeline definition for DVC
# Run with: dvc repro (when DVC is installed)
stages:
  data_ingestion:
    cmd: python pipelines/01_data_ingestion.py
    outs:
      - data/raw/drug_trials_train.csv
      - data/raw/drug_trials_production.csv

  data_validation:
    cmd: python pipelines/02_data_validation.py
    deps:
      - data/raw/drug_trials_train.csv
    outs:
      - data/processed/validation_report.json

  feature_engineering:
    cmd: python pipelines/03_feature_engineering.py
    deps:
      - data/raw/drug_trials_train.csv
      - data/raw/drug_trials_production.csv
    outs:
      - data/processed/features_train.csv
      - data/processed/reference.csv
      - data/processed/features_production.csv

  model_training:
    cmd: python pipelines/04_model_training.py
    deps:
      - data/processed/features_train.csv
    outs:
      - models/trained/best_model.joblib
      - models/trained/model_metadata.json
    metrics:
      - models/trained/model_metadata.json:
          cache: false

  model_validation:
    cmd: python pipelines/05_model_validation.py
    deps:
      - models/trained/best_model.joblib
      - data/processed/features_production.csv
    outs:
      - data/processed/qualification_report.json

  model_registry:
    cmd: python pipelines/06_model_registry.py
    deps:
      - models/trained/best_model.joblib
      - data/processed/qualification_report.json
    outs:
      - models/registry/current_production.json
"""
    Path("dvc.yaml").write_text(dvc_yaml)
    logger.info("  Created: dvc.yaml (pipeline definition)")
    logger.info("\nManual DVC config created. Install DVC to use full features:")
    logger.info("  pip install dvc  →  then re-run python dvc_setup.py")


if __name__ == "__main__":
    setup_dvc()
