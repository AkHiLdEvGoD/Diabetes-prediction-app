import json
import pandas as pd
from pathlib import Path
from src.utils.paths import RAW_DATA_DIR,INGESTION_STATE_FILE,VALIDATED_DIR,VALIDATED_STATE_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

NULL_THRESHOLD = 0.05
DUPLICATE_THRESHOLD = 0.01

def load_ingestion_state():
    if not INGESTION_STATE_FILE.exists():
        raise RuntimeError("Ingestion state not found")

    with open(INGESTION_STATE_FILE,"r") as f:
        return json.load(f)

def write_validation_state(status:str,checked_batch,errors:list):
    VALIDATED_STATE_FILE.parent.mkdir(parents=True,exist_ok=True)

    payload = {
        'status':status,
        "checked_batch":checked_batch,
        "error":errors
    }

    with open(VALIDATED_STATE_FILE,"w") as f:
        json.dump(payload,f,indent=2)

    logger.info(f'Validation status:{status}')


def run_quality_checks(df:pd.DataFrame,name:str,errors:list):
    if df.empty:
        errors.append(f"{name} dataset is empty")

    if df.shape[1]<2:
        errors.append(f"{name} has too few columns")

    for col in df.columns:
        if df[col].isnull().mean() > NULL_THRESHOLD:
            errors.append(f"{name}: null % exceeded for column '{col}'")

    dup_ratio = df.duplicated().mean()
    if dup_ratio > DUPLICATE_THRESHOLD:
        errors.append(f'{name}: duplicate row ratio {dup_ratio:.2%} exceeded')

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        errors.append(f"{name}: constant columns found {constant_cols}")

def run_validation():
    errors=[]
    checked_batch = None

    try:
        state = load_ingestion_state()
        baseline_path = RAW_DATA_DIR/"baseline.csv"
        if not baseline_path.exists():
            errors.append('Baseline file missing')
            write_validation_state("FAIL",None,errors)
            raise RuntimeError("Baseline missing")
        
        baseline_df = pd.read_csv(baseline_path)
        run_quality_checks(baseline_df,"Baseline",errors)

        if state['batch_index'] is None:
            if errors:
                write_validation_state("FAIL",None,errors)
                raise RuntimeError('Baseline validation failed')

            VALIDATED_DIR.mkdir(parents=True,exist_ok=True)
            baseline_df.to_csv(VALIDATED_DIR/"baseline.csv",index=False)
            write_validation_state("PASS",None,[])
            logger.success("Baseline-only validation passed")
            return

        batch_file = f"batch_{state['batch_index']+1:02d}.csv"
        batch_path = RAW_DATA_DIR/batch_file
        checked_batch = batch_file

        if not batch_path.exists():
            errors.append("Batch file missing")
            write_validation_state("FAIL", checked_batch, errors)
            raise RuntimeError("Batch missing")

        batch_df = pd.read_csv(batch_path)
        run_quality_checks(batch_df, "Batch", errors)

        if set(baseline_df.columns) != set(batch_df.columns):
            errors.append("Schema mismatch between baseline and batch")

        if not baseline_df.dtypes.equals(batch_df.dtypes):
            errors.append("Datatype mismatch between baseline and batch")

        if errors:
            write_validation_state("FAIL", checked_batch, errors)
            raise RuntimeError("Batch validation failed")

        VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
        baseline_df.to_csv(VALIDATED_DIR / "baseline.csv", index=False)
        batch_df.to_csv(VALIDATED_DIR / batch_file, index=False)

        write_validation_state("PASS", checked_batch, [])
        logger.success(f"Validation passed for {batch_file}")

    except Exception:
        logger.exception("Validation stage failed")
        raise

if __name__ == "__main__":
    run_validation()