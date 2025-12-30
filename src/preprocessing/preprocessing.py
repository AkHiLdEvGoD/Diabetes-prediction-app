import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from src.utils.paths import DATA_DIR,VALIDATED_DIR, DRIFT_STATE_FILE, REFERENCE_FILE, PROCESSED_DIR, LINEAR_DIR, TREE_DIR, CAT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COL = "diagnosed_diabetes"
DROP_COLS = ["id","diagnosed_diabetes"]

NUMERIC_FEATURES = [
    "age","alcohol_consumption_per_week","physical_activity_minutes_per_week",
    "diet_score","sleep_hours_per_day","screen_time_hours_per_day","bmi",
    "waist_to_hip_ratio","systolic_bp","diastolic_bp","heart_rate",
    "cholesterol_total","hdl_cholesterol","ldl_cholesterol","triglycerides"
]

CATEGORICAL_FEATURES = [
    "gender","ethnicity","education_level","income_level","smoking_status",
    "employment_status","family_history_diabetes",
    "hypertension_history","cardiovascular_history"
]

ENGINEERED_FEATURES = ["bmi_category", "activity_level"]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["underweight", "normal", "overweight", "obese"]
    )

    df["activity_level"] = pd.cut(
        df["physical_activity_minutes_per_week"],
        bins=[0, 75, 150, float("inf")],
        labels=["low", "moderate", "high"]
    )

    return df

def run_preprocessing():
    logger.info("Starting Preprocessing...")
    try:
        with open(DRIFT_STATE_FILE,"r") as f:
            drift_state = json.load(f)

        if drift_state['status']=='NO_DRIFT':
            logger.info("No drift detected. Skipping preprocessing")
            return

        if REFERENCE_FILE.exists():
            reference_df = pd.read_csv(REFERENCE_FILE)
            logger.info("Loaded cumulative reference data")
        else:
            reference_df = pd.read_csv(VALIDATED_DIR / "baseline.csv")
            logger.info("Loaded baseline as reference")

        if drift_state['status']=='DRIFT_DETECTED':
            batch_df = pd.read_csv(VALIDATED_DIR/drift_state['checked_batch'])
            df = pd.concat([reference_df,batch_df],ignore_index=True)
            logger.info(f"Preprocessing reference + {drift_state['checked_batch']}")
        else:
            # BOOTSTRAP case
            df = reference_df.copy()
            logger.info("Preprocessing baseline/reference only (bootstrap)")
        
        df = add_engineered_features(df)
        X=df.drop(columns=DROP_COLS)
        y=df[TARGET_COL]
        cat_features_all = CATEGORICAL_FEATURES + ENGINEERED_FEATURES
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        #Linear Model
        scaler = StandardScaler()
        ohe = OneHotEncoder(handle_unknown="ignore",sparse_output=False)   
        X_train_num = scaler.fit_transform(X_train[NUMERIC_FEATURES])
        X_test_num = scaler.transform(X_test[NUMERIC_FEATURES])
        X_train_cat = ohe.fit_transform(X_train[cat_features_all])
        X_test_cat = ohe.transform(X_test[cat_features_all])
        X_train_linear = pd.DataFrame(
            data=np.hstack([X_train_num, X_train_cat])
        )
        X_test_linear = pd.DataFrame(
            data=np.hstack([X_test_num, X_test_cat])
        )
        LINEAR_DIR.mkdir(parents=True,exist_ok=True)
        X_train_linear.to_csv(LINEAR_DIR / "X_train.csv", index=False)
        X_test_linear.to_csv(LINEAR_DIR / "X_test.csv", index=False)
        y_train.to_csv(LINEAR_DIR / "y_train.csv", index=False)
        y_test.to_csv(LINEAR_DIR / "y_test.csv", index=False)

        #TREE MODEL
        ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_tree = X_train.copy()
        X_test_tree = X_test.copy()
        X_train_tree[cat_features_all] = ordinal.fit_transform(X_train[cat_features_all])
        X_test_tree[cat_features_all] = ordinal.transform(X_test[cat_features_all])
        TREE_DIR.mkdir(parents=True,exist_ok=True)
        X_train_tree.to_csv(TREE_DIR / "X_train.csv", index=False)
        X_test_tree.to_csv(TREE_DIR / "X_test.csv", index=False)
        y_train.to_csv(TREE_DIR / "y_train.csv", index=False)
        y_test.to_csv(TREE_DIR / "y_test.csv", index=False)

        #CATBOOST
        catboost_train = X_train.copy()
        catboost_train[TARGET_COL] = y_train
        catboost_test = X_test.copy()
        catboost_test[TARGET_COL] = y_test
        CAT_DIR.mkdir(parents=True,exist_ok=True)
        catboost_train.to_csv(CAT_DIR / "train.csv", index=False)
        catboost_test.to_csv(CAT_DIR / "test.csv", index=False)

        feature_metadata = {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "engineered_features": ENGINEERED_FEATURES,
            "target": TARGET_COL
        }
        with open(DATA_DIR / "feature_metadata.json", "w") as f:
            json.dump(feature_metadata, f, indent=2)
        logger.success("Preprocessing completed for all model families")

    except Exception:
        logger.exception("Preprocessing failed")
        raise

if __name__ == "__main__":
    run_preprocessing()