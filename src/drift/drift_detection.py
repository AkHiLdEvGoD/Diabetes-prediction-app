import json
from evidently.presets.drift import DataDriftPreset
import pandas as pd
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset
from src.utils.paths import DRIFT_STATE_FILE,VALIDATED_DIR,REFERENCE_DIR,REFERENCE_FILE
from src.utils.logger import get_logger, logger

logger = get_logger(__name__)

DRIFT_THRESHOLD = 0.3
EXCLUDE_COLS = [
    "id",
    "diagnosed_diabetes"
]
CRITICAL_FEATURES = [
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "physical_activity_minutes_per_week"
]


def write_drift_state(payload:dict):
    DRIFT_STATE_FILE.parent.mkdir(parents=True,exist_ok=True)

    with open(DRIFT_STATE_FILE,"w") as f:
        json.dump(payload,f,indent=2)
    logger.info(f"Drift status:{payload.get('status')}")

def run_drift_detection():
    logger.info("Starting Drift Detection...")
    try:
        if REFERENCE_FILE.exists():
            reference_path = REFERENCE_FILE
            logger.info("Using cumulative reference data for drift detection")
        else:
            reference_path = VALIDATED_DIR / "baseline.csv"
            logger.info("Using baseline as reference data for drift detection")

        if not reference_path.exists():
            raise RuntimeError("Reference data not found")

        batch_files = sorted(VALIDATED_DIR.glob("batch_*.csv"))
        if not batch_files:
            payload={
                "status":"SKIPPED",
                "checked_batch":None,
                "reason":"No batch available for drift detection"
            }
            write_drift_state(payload)
            logger.info("Drift detection skipped (baseline-only)")
            return
        batch_path = batch_files[-1]
        logger.info(f"Running drift detection for {batch_path.name}")
        
        reference_df = pd.read_csv(reference_path)
        batch_df = pd.read_csv(batch_path)

        reference_df = reference_df.drop(columns=EXCLUDE_COLS, errors="ignore")
        batch_df = batch_df.drop(columns=EXCLUDE_COLS, errors="ignore")

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(
            reference_data = reference_df,
            current_data = batch_df
        )
        report_dict = snapshot.dict()
        metrics = report_dict["metrics"]
        drift_count_metric = next(
            m for m in metrics
            if m.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount"
        )

        drifted_columns = drift_count_metric["value"]["count"]
        drift_score = drift_count_metric["value"]["share"]

        column_drift = {}

        for m in metrics:
            if m.get("config", {}).get("type") == "evidently:metric_v2:ValueDrift":
                col = m["config"]["column"]
                threshold = m["config"]["threshold"]
                drift_value = m["value"]

                column_drift[col] = {
                    "drift_value": drift_value,
                    "threshold": threshold,
                    "drift_detected": drift_value >= threshold
                }

        critical_drifted_features = [
            col for col in CRITICAL_FEATURES
            if col in column_drift and column_drift[col]["drift_detected"]
        ]
        critical_drift = len(critical_drifted_features) > 0

        if drift_score >= DRIFT_THRESHOLD or critical_drift:
            status = "DRIFT_DETECTED"
        else:
            status = "NO_DRIFT"

        payload = {
            "status": status,
            "checked_batch": batch_path.name,
            "drift_score": round(drift_score, 4),
            "drifted_columns": int(drifted_columns),
            "threshold": DRIFT_THRESHOLD,
            "critical_drift": critical_drift,
            "critical_features_drifted": critical_drifted_features,
            "reference_used": reference_path.name
        }
        write_drift_state(payload)
        if status == "DRIFT_DETECTED":
            logger.warning(
                f"Drift detected | "
                f"reference={reference_path.name}, "
                f"global_score={drift_score:.2f}, "
                f"critical_features={critical_drifted_features}"
            )
        else:
            logger.success("No significant drift detected")

    except Exception:
        logger.exception("Drift detection failed")
        raise

if __name__ == "__main__":
    run_drift_detection()