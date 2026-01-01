import shutil
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from mlflow import MlflowClient
from src.utils.logger import get_logger
from src.utils.paths import MODEL_METRICS_FILE,PROMOTION_STATE_FILE, REFERENCE_DIR,REFERENCE_FILE,VALIDATED_DIR,DRIFT_STATE_FILE

logger = get_logger(__name__)

MODEL_NAME_IN_REGISTRY = "diabetes_detection_model"
def load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"{path.name} not found")
    with open(path, "r") as f:
        return json.load(f)

def promote_model(run_id:str):
    client=MlflowClient()
    versions=client.get_latest_versions(
        name=MODEL_NAME_IN_REGISTRY,
        stages=['Production']
    )
    if versions:
        for v in versions:
            client.transition_model_version_stage(
                name=MODEL_NAME_IN_REGISTRY,
                versions=v.version,
                stage="Archived"
            )
        logger.info(f"Archived previous Production model v{v.version}")
    else:
        logger.info("No existing Production model to archive")

    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME_IN_REGISTRY}' and run_id='{run_id}'"
    )

    if not versions:
        raise RuntimeError(
            f"No model version found for run_id={run_id}. "
            "Model must be registered during training."
        )
    mv = versions[0]
    client.transition_model_version_stage(
        name=MODEL_NAME_IN_REGISTRY,
        version=mv.version,
        stage='Production'
    )
    logger.success(
        f"Promoted model run {run_id} → Production (v{mv.version})"
    )

def update_reference_data(promotion:dict):
    logger.info('Updating reference data...')
    REFERENCE_DIR.mkdir(parents=True,exist_ok=True)
    drift = load_json(DRIFT_STATE_FILE)

    if not promotion.get("promote"):
        raise RuntimeError(
            "update_reference_data called without model promotion"
        )
    
    if drift["status"] == "SKIPPED":
        src = VALIDATED_DIR / "baseline.csv"
        shutil.copy(src, REFERENCE_FILE)
        logger.success("Reference initialized from baseline.csv")
        return

    if drift["status"] != "DRIFT_DETECTED":
        logger.info("No drift detected. Reference not updated.")
        return

    checked_batch = drift.get("checked_batch")
    if not checked_batch:
        raise RuntimeError("checked_batch missing in drift_state.json")

    old_ref = REFERENCE_FILE
    new_batch = VALIDATED_DIR / checked_batch
    if not old_ref.exists():
        raise RuntimeError("Existing reference.csv not found")

    df_ref = pd.read_csv(old_ref)
    df_batch = pd.read_csv(new_batch)

    updated_ref = pd.concat([df_ref, df_batch], ignore_index=True)
    updated_ref.to_csv(REFERENCE_FILE, index=False)

    logger.success(f"Reference updated with {checked_batch}")

def run_model_registry():
    logger.info("Starting model registry...")
    try:
        promotion = load_json(PROMOTION_STATE_FILE)
        if not promotion.get("promote"):
            logger.info("Model not promoted. Skipping model registry.")
            return
        metrics = load_json(MODEL_METRICS_FILE)
        run_id = metrics["run_id"]
        promote_model(run_id)
        update_reference_data(promotion)
    except Exception as e:
        logger.error(f"Failed to run model promotion: {str(e)}")
        raise

if __name__ == "__main__":
    run_model_registry()