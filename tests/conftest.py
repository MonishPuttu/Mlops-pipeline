# tests/conftest.py
# pytest configuration — sets up importable aliases for numbered pipeline modules

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _register(path: str, alias: str):
    """Load a file as a module and register it under `alias` in sys.modules."""
    if alias in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod          # register BEFORE exec to handle circular refs
    spec.loader.exec_module(mod)


# Make each numbered pipeline stage importable as pipelines._01_* etc.
_ALIASES = {
    "pipelines._01_data_ingestion":       "pipelines/01_data_ingestion.py",
    "pipelines._02_data_validation":      "pipelines/02_data_validation.py",
    "pipelines._03_feature_engineering":  "pipelines/03_feature_engineering.py",
    "pipelines._04_model_training":       "pipelines/04_model_training.py",
    "pipelines._05_model_validation":     "pipelines/05_model_validation.py",
    "pipelines._06_model_registry":       "pipelines/06_model_registry.py",
}

for alias, rel_path in _ALIASES.items():
    _register(str(ROOT / rel_path), alias)
