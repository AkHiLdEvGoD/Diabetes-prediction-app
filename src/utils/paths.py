from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR  = PROJECT_ROOT/'data'
RAW_DATA_DIR = DATA_DIR / "raw"
INGESTION_STATE_FILE = DATA_DIR / "ingestion_state.json"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
INGESTION_CONFIG = CONFIG_DIR / "ingestion.yaml"
VALIDATED_DIR = DATA_DIR / "validated"
VALIDATED_STATE_FILE = DATA_DIR/"validation_state.json"
DRIFT_STATE_FILE = DATA_DIR/"drift_state.json"
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_FILE = REFERENCE_DIR / "reference.csv"
PROCESSED_DIR = DATA_DIR / "processed"
LINEAR_DIR = PROCESSED_DIR/"linear"
TREE_DIR = PROCESSED_DIR/"tree"
CAT_DIR = PROCESSED_DIR/"catboost"
MODEL_METRICS_FILE = DATA_DIR / "model_metrics.json"
PROMOTION_STATE_FILE = DATA_DIR / "promotion_state.json"