from dotenv import load_dotenv
load_dotenv()
import os
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.utils.paths import DATA_DIR,LINEAR_DIR,TREE_DIR,CAT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
REGISTERED_MODEL_NAME = "diabetes_detection_model"
EXPERIMENT_NAME = "diabetes_detection_models"
MODEL_FAMILY_MAP = {
    "logreg": "linear",
    "rf": "tree",
    "xgb": "tree",
    "catboost": "boosting"
}
PRIMARY_METRIC = "roc_auc"
N_TRIALS = 25
with open(DATA_DIR / "feature_metadata.json",'r') as f:
    feature_meta = json.load(f)
CATBOOST_CAT_FEATURES = (feature_meta['categorical_features']+feature_meta['engineered_features'])

def load_xy(path: Path):
    X_train = pd.read_csv(path / "X_train.csv")
    X_test = pd.read_csv(path / "X_test.csv")
    y_train = pd.read_csv(path / "y_train.csv").squeeze()
    y_test = pd.read_csv(path / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def load_catboost_data(path: Path, target_col: str):
    train_df = pd.read_csv(path / "train.csv")
    test_df = pd.read_csv(path / "test.csv")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

def objective_catboost(trial, X_train, X_test, y_train, y_test, cat_features_idx):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "silent": True,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features_idx,
        eval_set=(X_test, y_test),
        verbose=False
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

def objective_logreg(trial, X_train, X_test, y_train, y_test):
    params = {
        "C": trial.suggest_float("C", 1e-3, 10, log=True),
        "solver": "liblinear",
        "max_iter": trial.suggest_int("max_iter",500,1000),
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

def objective_xgb(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eval_metric": "auc",
        "use_label_encoder": False,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)])
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

def train_logreg():
    X_train, X_test, y_train, y_test = load_xy(LINEAR_DIR)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective_logreg(t, X_train, X_test, y_train, y_test),
        n_trials=N_TRIALS,
        timeout=600,
    )

    best_params = study.best_params

    with mlflow.start_run(run_name="logreg") as run:
        model = LogisticRegression(**best_params, solver="liblinear")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

    return "logreg", model, MODEL_FAMILY_MAP["logreg"], metrics, run.info.run_id


def train_xgb():
    X_train, X_test, y_train, y_test = load_xy(TREE_DIR)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective_xgb(t, X_train, X_test, y_train, y_test),
        n_trials=N_TRIALS,
        timeout=600,
    )

    best_params = study.best_params

    with mlflow.start_run(run_name="xgb") as run:
        model = XGBClassifier(**best_params, eval_metric="auc", use_label_encoder=False,verbose=False)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

    return "xgb", model, MODEL_FAMILY_MAP["xgb"], metrics, run.info.run_id


def train_catboost():
    X_train, X_test, y_train, y_test = load_catboost_data(
        CAT_DIR, target_col="diagnosed_diabetes"
    )

    cat_idx = [X_train.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES]

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective_catboost(t, X_train, X_test, y_train, y_test, cat_idx),
        n_trials=N_TRIALS,
        timeout=600,
    )

    best_params = study.best_params

    with mlflow.start_run(run_name="catboost") as run:
        model = CatBoostClassifier(
            **best_params,
            loss_function="Logloss",
            eval_metric="AUC",
            allow_writing_files=False,
            silent=True
        )
        model.fit(X_train, y_train, cat_features=cat_idx)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

    return "catboost", model, MODEL_FAMILY_MAP["catboost"], metrics, run.info.run_id


# Orchestration
def run_training():
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("Starting training for all candidate models")

    results = []
    results.append(train_logreg())
    results.append(train_xgb())
    results.append(train_catboost())

    # Select best model
    best = max(results, key=lambda x: x[3][PRIMARY_METRIC])
    model_name, model_obj, model_family, metrics, run_id = best

    logger.success(
        f"Best model: {model_name} | ROC-AUC={metrics['roc_auc']:.4f}"
    )
    logger.info(f"Registering bestmodel {model_name} with run ID {run_id}")
    with mlflow.start_run(run_id=run_id):
        if model_name == "catboost":
            mlflow.catboost.log_model(
                model_obj,
                name=f"{model_name}_model",
                registered_model_name=REGISTERED_MODEL_NAME
            )
        else:
            mlflow.sklearn.log_model(
                model_obj,
                name=f"{model_name}_model",
                registered_model_name=REGISTERED_MODEL_NAME
            )
    logger.success(f"Model {model_name} registered successfully")
    # ------------------
    # Persist metadata
    # ------------------
    payload = {
        "model_name": model_name,
        "model_family": model_family,
        "run_id": run_id,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    with open(DATA_DIR / "model_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    logger.success("Training complete. Best model registered.")


if __name__ == "__main__":
    run_training()