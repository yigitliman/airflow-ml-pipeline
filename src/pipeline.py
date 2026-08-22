"""
Core pipeline steps called by the Airflow DAGs.

Each function is a standalone step that reads from and writes to a per-run
artifact directory. This keeps the DAGs clean and the logic testable.

Artifact layout:

    data/reference.csv          shared, long-lived drift baseline
    data/runs/<run_id>/         everything belonging to one pipeline run

Per-run namespacing matters because two DAGs share this module: the daily
training pipeline and the 6-hourly drift monitor both call fetch_data. With a
single data/raw.csv they raced -- the monitor could overwrite the training
run's input mid-flight. Only the reference snapshot stays shared, since it is
by definition the one baseline both DAGs compare against, and train_model is
its single writer.
"""

import hashlib
import json
import os
import re
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ.get(
    "PIPELINE_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
)

# The library default. The production DAG deliberately runs a stricter gate
# (0.72); see the README for why the two differ.
DEFAULT_MIN_ROC_AUC = 0.70

# Used when a step is called outside Airflow (tests, `python -m`, manual runs).
DEFAULT_RUN_ID = "manual"

FEATURES = ["tenure", "monthly_charges", "num_products",
            "has_internet", "support_calls", "contract_months"]


def _safe_run_id(run_id: str) -> str:
    """Airflow run ids contain ':' and '+'; make one safe as a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(run_id)) or DEFAULT_RUN_ID


def _seed_from(value: str) -> int:
    """Stable 32-bit seed derived from a string.

    Python's built-in hash() is salted per process, so it would give a
    different answer on every worker; sha256 does not.
    """
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16)


def run_paths(run_id: str = DEFAULT_RUN_ID, data_dir: str | None = None) -> dict:
    """Resolve every path one run touches.

    Pass `data_dir` to redirect the whole layout (tests use tmp_path).
    """
    base = data_dir or DATA_DIR
    run_dir = os.path.join(base, "runs", _safe_run_id(run_id))
    return {
        "data_dir": base,
        "run_dir": run_dir,
        "raw": os.path.join(run_dir, "raw.csv"),
        "processed": os.path.join(run_dir, "processed.csv"),
        "scaler": os.path.join(run_dir, "scaler.joblib"),
        "model": os.path.join(run_dir, "model.joblib"),
        "report": os.path.join(run_dir, "report.json"),
        "run_log": os.path.join(run_dir, "run_log.json"),
        "drift_report": os.path.join(run_dir, "drift_report.json"),
        # Shared and long-lived: NOT namespaced by run.
        "reference": os.path.join(base, "reference.csv"),
    }


def _append_run_log(paths: dict, step: str, status: str, details: dict = None):
    """Append one entry to this run's log.

    Rotation strategy: the log is *scoped per run id* rather than capped. One
    run writes a handful of entries and then never touches the file again, so
    it is bounded by construction -- the old single shared run_log.json grew
    without limit and was read-modify-written concurrently by two DAGs.
    """
    os.makedirs(paths["run_dir"], exist_ok=True)
    logs = []
    if os.path.exists(paths["run_log"]):
        with open(paths["run_log"]) as f:
            logs = json.load(f)
    logs.append({
        "step": step,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(details or {}),
    })
    with open(paths["run_log"], "w") as f:
        json.dump(logs, f, indent=2)


def fetch_data(
    n_samples: int = 2000,
    seed: int | None = None,
    logical_date: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    data_dir: str | None = None,
) -> None:
    """Generate a synthetic customer batch for this run.

    This is a stand-in for a real source system, not a real extract.

    When `seed` is not given it is derived from the run's logical date (or
    failing that, the run id). A fixed default seed meant the "daily" pipeline
    regenerated a byte-identical dataset every single day and never ingested
    anything new.
    """
    paths = run_paths(run_id, data_dir)
    if seed is None:
        seed = _seed_from(logical_date or run_id)
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n_samples),
        "monthly_charges": rng.uniform(20, 120, n_samples).round(2),
        "num_products": rng.integers(1, 6, n_samples),
        "has_internet": rng.integers(0, 2, n_samples),
        "support_calls": rng.integers(0, 10, n_samples),
        "contract_months": rng.choice([1, 12, 24], n_samples),
    })

    churn_prob = (
        0.05
        + 0.30 * (df["contract_months"] == 1)
        + 0.15 * (df["monthly_charges"] > 80)
        - 0.10 * (df["tenure"] > 24)
        + 0.05 * (df["support_calls"] > 5)
    ).clip(0.02, 0.95)

    df["churn"] = rng.binomial(1, churn_prob).astype(int)

    os.makedirs(paths["run_dir"], exist_ok=True)
    df.to_csv(paths["raw"], index=False)
    _append_run_log(paths, "fetch_data", "success", {"rows": n_samples, "seed": seed})
    print(f"Fetched {len(df)} rows (seed {seed}) -> {paths['raw']}")


def preprocess_data(run_id: str = DEFAULT_RUN_ID, data_dir: str | None = None) -> None:
    paths = run_paths(run_id, data_dir)
    df = pd.read_csv(paths["raw"])

    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    df.to_csv(paths["processed"], index=False)
    joblib.dump(scaler, paths["scaler"])
    _append_run_log(paths, "preprocess_data", "success", {"features": FEATURES})
    print(f"Preprocessed data -> {paths['processed']}")


def train_model(
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    run_id: str = DEFAULT_RUN_ID,
    data_dir: str | None = None,
) -> None:
    paths = run_paths(run_id, data_dir)
    df = pd.read_csv(paths["processed"])
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("airflow-churn-pipeline")
    with mlflow.start_run():
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        mlflow.log_metrics(metrics)

        joblib.dump({"model": model, "metrics": metrics, "params": params}, paths["model"])

        # Snapshot the raw training features as the shared drift reference.
        # train_model is the only writer of this file.
        pd.read_csv(paths["raw"]).drop(columns=["churn"]).to_csv(paths["reference"], index=False)

        _append_run_log(paths, "train_model", "success", {"metrics": metrics})
        print(f"Model trained. Metrics: {metrics}")


def evaluate_model(
    min_roc_auc: float = DEFAULT_MIN_ROC_AUC,
    run_id: str = DEFAULT_RUN_ID,
    data_dir: str | None = None,
) -> dict:
    paths = run_paths(run_id, data_dir)
    bundle = joblib.load(paths["model"])
    metrics = bundle["metrics"]

    passed = metrics["roc_auc"] >= min_roc_auc
    report = {
        "status": "pass" if passed else "fail",
        "metrics": metrics,
        "quality_gate": {"min_roc_auc": min_roc_auc},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(paths["report"], "w") as f:
        json.dump(report, f, indent=2)

    _append_run_log(paths, "evaluate_model", report["status"], {"roc_auc": metrics["roc_auc"]})
    print(f"Evaluation: {report['status']} (ROC-AUC: {metrics['roc_auc']})")

    if not passed:
        raise ValueError(
            f"Model failed quality gate. ROC-AUC: {metrics['roc_auc']:.4f} < {min_roc_auc}"
        )
    return report


def check_drift(
    current_path: str | None = None,
    p_threshold: float = 0.05,
    drift_share: float = 0.5,
    run_id: str = DEFAULT_RUN_ID,
    data_dir: str | None = None,
) -> dict:
    """Compare the latest data batch against the training reference.

    Runs a Kolmogorov-Smirnov test per feature. The dataset is considered
    drifted when at least `drift_share` of the features reject the null
    hypothesis at `p_threshold`. Defaults to this run's raw batch.
    """
    paths = run_paths(run_id, data_dir)
    current_path = current_path or paths["raw"]

    if not os.path.exists(paths["reference"]):
        report = {"drift_detected": False, "reason": "no_reference_data"}
        _append_run_log(paths, "check_drift", "skipped", report)
        return report

    reference = pd.read_csv(paths["reference"])
    current = pd.read_csv(current_path)
    features = list(reference.columns)

    drifted = []
    p_values = {}
    for feature in features:
        _, p_value = ks_2samp(reference[feature], current[feature])
        p_values[feature] = round(float(p_value), 6)
        if p_value < p_threshold:
            drifted.append(feature)

    drift_detected = len(drifted) / len(features) >= drift_share
    report = {
        "drift_detected": drift_detected,
        "drifted_features": drifted,
        "p_values": p_values,
        "thresholds": {"p_threshold": p_threshold, "drift_share": drift_share},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(paths["drift_report"], "w") as f:
        json.dump(report, f, indent=2)

    _append_run_log(paths, "check_drift", "drift" if drift_detected else "no_drift",
                    {"drifted_features": drifted})
    print(f"Drift detected: {drift_detected} (drifted: {drifted})")
    return report


def run_pipeline(
    *,
    n_samples: int = 2000,
    min_roc_auc: float = DEFAULT_MIN_ROC_AUC,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    seed: int | None = None,
    run_id: str = DEFAULT_RUN_ID,
    data_dir: str | None = None,
) -> dict:
    paths = run_paths(run_id, data_dir)
    if os.path.exists(paths["run_log"]):
        os.remove(paths["run_log"])

    fetch_data(n_samples=n_samples, seed=seed, run_id=run_id, data_dir=data_dir)
    preprocess_data(run_id=run_id, data_dir=data_dir)
    train_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        run_id=run_id,
        data_dir=data_dir,
    )
    return evaluate_model(min_roc_auc=min_roc_auc, run_id=run_id, data_dir=data_dir)
