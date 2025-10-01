# config.py
# Portable configuration for AutoEnergy.
# Paths default to the repository directory (where this file lives).
# Users can override the base directory by setting:
#   AUTOENERGY_BASE_DIR=/absolute/path/to/project
# Optionally override processors with:
#   AUTOENERGY_PROCESSORS="AutoEnergy,No_Feat_Eng"

from pathlib import Path
import os

# --- Base directory ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("AUTOENERGY_BASE_DIR", str(REPO_ROOT)))

# --- I/O directories --------------------------------------------------------
# Kept as strings for compatibility with code using os.path.*
INPUT_DIR = str(BASE_DIR / "data" / "raw")               # where *.csv files live :)
PREPROCESSED_DIR = str(BASE_DIR / "data" / "processed")  # where stage1 writes parquet splits
RESULTS_DIR = str(BASE_DIR / "results")                  # where metrics/outputs go
IMPORTANCE_DIR = str(BASE_DIR / "feature_importance")    # where feature importance is stored
LOG_DIR = str(BASE_DIR / "logs")                         # where log files are written

# --- Processing configuration ----------------------------------------------
# Must match the names expected by utils.preprocess_and_split(...)
_default_processors = [
    "AutoEnergy",
    "Featuretools",
    "TSfresh_MinimalFCParameters",
    "TSfresh_EfficientFCParameters",
    "TSfresh_ComprehensiveFCParameters",
    "No_Feat_Eng",
]

# Allow override via env var AUTOENERGY_PROCESSORS (comma-separated)
_env_procs = os.environ.get("AUTOENERGY_PROCESSORS")
PROCESSOR_TYPES = (
    [p.strip() for p in _env_procs.split(",") if p.strip()]
    if _env_procs else _default_processors
)

# --- Utility ---------------------------------------------------------------
def ensure_directories() -> None:
    """
    Create all required directories if they do not exist.
    Safe to call multiple times.
    """
    for p in [INPUT_DIR, PREPROCESSED_DIR, RESULTS_DIR, IMPORTANCE_DIR, LOG_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)

