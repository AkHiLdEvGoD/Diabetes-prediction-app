from dotenv import load_dotenv
import os
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.utils.paths import DATA_DIR,LINEAR_DIR,TREE_DIR,CAT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

EXPERIMENT_NAME = "diabetes_prediction_training"
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

def log_metrics(y_true, y_pred, y_prob):
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
        "verbose": False
    }
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features_idx,
        eval_set=(X_test, y_test),
        verbose=False
    )
    trial.report(model.get_best_score()["validation"]["AUC"], step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()
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

def train_and_log(model_name, objective_fn, data_path):
    X_train, X_test, y_train, y_test = load_xy(data_path)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=10))
    study.optimize(
        lambda t: objective_fn(t, X_train, X_test, y_train, y_test),
        n_trials=N_TRIALS,
        timeout=600
    )

    best_params = study.best_params

    with mlflow.start_run(run_name=model_name):
        if model_name == "logreg":
            model = LogisticRegression(**best_params, solver="liblinear")
        elif model_name == "xgb":
            model = XGBClassifier(**best_params, eval_metric="auc", use_label_encoder=False)
        else:
            raise ValueError("Unknown model")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = log_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.success(f"{model_name} | ROC-AUC={metrics['roc_auc']:.4f}")

        return model_name, MODEL_FAMILY_MAP[model_name], metrics

def train_catboost(path:str):
    X_train, X_test, y_train, y_test = load_catboost_data(
        path,
        target_col="diagnosed_diabetes"
    )

    cat_features_idx = [
        X_train.columns.get_loc(col)
        for col in CATBOOST_CAT_FEATURES
    ]

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=10))
    study.optimize(
        lambda t: objective_catboost(
            t, X_train, X_test, y_train, y_test, cat_features_idx
        ),
        n_trials=N_TRIALS,
        timeout=600
    )

    best_params = study.best_params

    with mlflow.start_run(run_name="catboost"):
        model = CatBoostClassifier(
            **best_params,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False
        )

        model.fit(
            X_train, y_train,
            cat_features=cat_features_idx,
            eval_set=(X_test, y_test),
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = log_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.catboost.log_model(model, artifact_path="model")

        logger.success(f"catboost | ROC-AUC={metrics['roc_auc']:.4f}")

        return "catboost", MODEL_FAMILY_MAP["catboost"], metrics

def run_training():

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []
    logger.info("Starting model training..")
    results.append(train_and_log("logreg", objective_logreg, LINEAR_DIR))
    results.append(train_and_log("xgb", objective_xgb, TREE_DIR))
    results.append(train_catboost(CAT_DIR))
    logger.success("All models trained and logged to DagsHub MLflow")

    best_model = max(results, key=lambda x: x[2]['roc_auc'])  # (model_name, roc_auc)

    metrics_payload = {
        "model_name": best_model[0],
        "model_family":best_model[1],
        "metrics": {
            "roc_auc": best_model[2]['roc_auc'],
            "accuracy":best_model[2]['accuracy'],
            "f1":best_model[2]['f1'],
            "recall":best_model[2]['recall'],
            "precision":best_model[2]['precision'],
        },
        "timestamp": datetime.now().isoformat()
    }
    with open(DATA_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    
if __name__ == "__main__":
    run_training()