import json
import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.utils.paths import MODEL_METRICS_FILE, PROMOTION_STATE_FILE
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

EXPERIMENT_NAME = "diabetes_detection_models"
MODEL_NAME_IN_REGISTRY = "diabetes_detection_model"
MIN_IMPROVEMENT = 0.005
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


def load_candidate_model():
    if not MODEL_METRICS_FILE.exists():
        logger.error("model_metrics.json does not exists")
        raise RuntimeError("model_metrics.json not found")
    with open(MODEL_METRICS_FILE,"r") as f:
        return json.load(f)

def get_production_model_metrics():
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(
            name = MODEL_NAME_IN_REGISTRY,
            stages=['Production']
        )
        if not versions:
            return None
        prod_version = versions[0]
        run = client.get_run(prod_version.run_id)

        return {
            "model_name": MODEL_NAME_IN_REGISTRY,
            "run_id": prod_version.run_id,
            "roc_auc": run.data.metrics.get("roc_auc")
        }

    except Exception:
        logger.warning("No production model found")
        return None
    
def write_promotion_state(payload: dict):
    with open(PROMOTION_STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Promotion decision: {payload['promote']}")

def run_evaluation():
    logger.info("Starting model evaluation...")
    candidate = load_candidate_model()
    production = get_production_model_metrics()
    candidate_auc = candidate['metrics']['roc_auc']

    if production is None:
        payload={
            "promote": True,
            "reason": "No production model exists (bootstrap)",
            "candidate": {
                "model_name": candidate["model_name"],
                "model_family": candidate["model_family"],
                "roc_auc": candidate_auc
            },
            "production":None,
            "timestamp":datetime.now().isoformat()
        }

        write_promotion_state(payload)
        logger.success("Bootstrap promotion approved")
        return

    prod_auc = production["roc_auc"]
    improvement = candidate_auc - prod_auc
    promote = improvement >= MIN_IMPROVEMENT

    payload = {
        "promote": promote,
        "reason": (
            f"Candidate ROC-AUC improved by {improvement:.4f}"
            if promote else
            f"Improvement {improvement:.4f} < threshold {MIN_IMPROVEMENT}"
        ),
        "candidate": {
            "model_name": candidate["model_name"],
            "model_family": candidate["model_family"],
            "roc_auc": candidate_auc
        },
        "production": {
            "model_name": production["model_name"],
            "roc_auc": prod_auc
        },
        "timestamp": datetime.now().isoformat()
    }

    write_promotion_state(payload)

    if promote:
        logger.success("Model promotion approved")
    else:
        logger.info("Model promotion rejected")

if __name__=="__main__":
    run_evaluation()



