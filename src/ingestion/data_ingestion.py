import yaml,json
from pathlib import Path
import boto3
from src.utils.paths import DATA_DIR, INGESTION_CONFIG, RAW_DATA_DIR,INGESTION_STATE_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_ingestion_config(config_path:Path)->dict:
    try:
        logger.info(f'Loading ingestion config from {config_path}')
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f'Ingestion config not found: {config_path}')
        raise 

def load_ingestion_state():
    try:
        if not INGESTION_STATE_FILE.exists():
            return {
                "baseline_injected": False,
                "batch_index": None
            }

        with open(INGESTION_STATE_FILE, "r") as f:
            return json.load(f)

    except Exception as e:
        logger.error('Unexpected error occured')
        raise 

def save_ingestion_state(state: dict):
    try:
        INGESTION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        relative_path = INGESTION_STATE_FILE.relative_to(DATA_DIR)
        with open(INGESTION_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logger.success(f'Saved ingestion_state file at {relative_path}')

    except Exception as e:
        logger.error('INGESTION_STATE_FILE cannot be saved')
        raise 

def download_from_s3(bucket: str, key: str, destination: Path, s3_client):
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative_path = destination.relative_to(DATA_DIR)
        logger.info(f"Downloading s3://{bucket}/{key} → {relative_path}")

        s3_client.download_file(bucket, key, str(destination))
        logger.success(f"Downloded s3://{bucket}/{key} → {relative_path}")
    except Exception as e:
        logger.error(f'Failed to download s3://{bucket}/{key}')
        raise 

def run_ingestion():
    try:
        config = load_ingestion_config(INGESTION_CONFIG)
        state = load_ingestion_state()
        source_cfg = config["data_source"]
        source_type = source_cfg["type"]

        if source_type != "s3":
            raise NotImplementedError(f"Unsupported data source: {source_type}")

        s3_cfg = source_cfg["s3"]
        bucket = s3_cfg["bucket"]
        prefixes = s3_cfg["prefixes"]

        s3_client = boto3.client("s3")

        if state["baseline_injected"]:
            logger.info('Baseline already ingested, proceeding to batch ingestion')
        else:
            baseline_file = source_cfg["files"]["baseline"]
            baseline_key = f"{prefixes['baseline'].rstrip('/')}/{baseline_file}"

            dest = RAW_DATA_DIR / baseline_file

            download_from_s3(bucket, baseline_key, dest, s3_client)

            state["baseline_injected"] = True
            logger.success("Baseline ingestion completed")
            save_ingestion_state(state)
            return

        batches = source_cfg["files"]["batches"]
        next_batch_idx = 0 if state["batch_index"] is None else state["batch_index"] + 1

        if next_batch_idx < len(batches):
            batch_file = batches[next_batch_idx]
            batch_key = f"{prefixes['batches'].rstrip('/')}/{batch_file}"

            dest = RAW_DATA_DIR / batch_file

            download_from_s3(bucket, batch_key, dest, s3_client)

            state["batch_index"] = next_batch_idx
            logger.success(f"Batch ingestion completed: {batch_file}")
        else:
            logger.info("No new batch available to ingest")
        save_ingestion_state(state)

    except Exception as e:
        logger.error('Unexpected error occured')
        raise 

if __name__=="__main__":
    run_ingestion()