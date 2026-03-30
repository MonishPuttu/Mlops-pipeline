# config/utils.py - Shared utilities for the entire pipeline

import yaml
import json
import logging
import os
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_logger(name: str) -> logging.Logger:
    cfg = load_config()
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def audit_log(event: str, details: dict, actor: str = "pipeline"):
    """Write a GxP-compliant immutable audit log entry."""
    cfg = load_config()
    audit_dir = Path(cfg["paths"]["audit"])
    audit_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "actor": actor,
        "event": event,
        "details": details,
        "system": cfg["project"]["name"],
        "version": cfg["project"]["version"],
    }

    log_file = audit_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry

def ensure_dirs():
    """Create all required directories."""
    cfg = load_config()
    for key, path in cfg["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)
