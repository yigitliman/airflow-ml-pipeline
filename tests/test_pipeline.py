import json
import os

import joblib
import pandas as pd
import pytest

from src.pipeline import (
    DATA_DIR,
    check_drift,
    evaluate_model,
    fetch_data,
    preprocess_data,
    run_paths,
    run_pipeline,
    train_model,
)


@pytest.fixture
def data_dir(tmp_path):
    """Every test gets its own artifact tree, so nothing lands in ./data."""
    return str(tmp_path / "data")


@pytest.fixture
def paths(data_dir):
    return run_paths("test-run", data_dir)


def test_fetch_data_creates_csv(data_dir, paths):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    assert os.path.exists(paths["raw"])
    df = pd.read_csv(paths["raw"])
    assert len(df) == 200
    assert "churn" in df.columns


def test_fetch_data_seed_varies_with_logical_date(data_dir):
    fetch_data(n_samples=200, logical_date="2024-01-01", run_id="day-1", data_dir=data_dir)
    fetch_data(n_samples=200, logical_date="2024-01-02", run_id="day-2", data_dir=data_dir)
    day1 = pd.read_csv(run_paths("day-1", data_dir)["raw"])
    day2 = pd.read_csv(run_paths("day-2", data_dir)["raw"])
    assert not day1.equals(day2)


def test_fetch_data_same_logical_date_is_reproducible(data_dir):
    fetch_data(n_samples=200, logical_date="2024-01-01", run_id="a", data_dir=data_dir)
    fetch_data(n_samples=200, logical_date="2024-01-01", run_id="b", data_dir=data_dir)
    first = pd.read_csv(run_paths("a", data_dir)["raw"])
    second = pd.read_csv(run_paths("b", data_dir)["raw"])
    assert first.equals(second)


def test_runs_are_isolated_from_each_other(data_dir):
    """The training DAG and the drift monitor must not share raw.csv."""
    fetch_data(n_samples=200, seed=1, run_id="training-run", data_dir=data_dir)
    fetch_data(n_samples=500, seed=2, run_id="drift-run", data_dir=data_dir)

    training = pd.read_csv(run_paths("training-run", data_dir)["raw"])
    drift = pd.read_csv(run_paths("drift-run", data_dir)["raw"])
    assert len(training) == 200
    assert len(drift) == 500


def test_airflow_style_run_id_is_path_safe(data_dir):
    run_id = "scheduled__2024-01-01T00:00:00+00:00"
    fetch_data(n_samples=100, run_id=run_id, data_dir=data_dir)
    assert os.path.exists(run_paths(run_id, data_dir)["raw"])


def test_preprocess_creates_csv(data_dir, paths):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    assert os.path.exists(paths["processed"])


def test_train_creates_model(data_dir, paths):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)
    assert os.path.exists(paths["model"])


def test_train_custom_hyperparams(data_dir, paths):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(n_estimators=50, max_depth=3, learning_rate=0.05,
                run_id="test-run", data_dir=data_dir)
    bundle = joblib.load(paths["model"])
    assert bundle["params"]["n_estimators"] == 50
    assert bundle["params"]["max_depth"] == 3


def test_evaluate_creates_report(data_dir, paths):
    fetch_data(n_samples=500, seed=1, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)
    report = evaluate_model(run_id="test-run", data_dir=data_dir)
    assert os.path.exists(paths["report"])
    assert "status" in report
    assert "metrics" in report
    assert "quality_gate" in report


def test_evaluate_fails_strict_threshold(data_dir):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)
    with pytest.raises(ValueError, match="quality gate"):
        evaluate_model(min_roc_auc=1.0, run_id="test-run", data_dir=data_dir)


def test_run_log_is_written(data_dir, paths):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    assert os.path.exists(paths["run_log"])
    with open(paths["run_log"]) as f:
        log = json.load(f)
    assert any(entry["step"] == "fetch_data" for entry in log)


def test_run_log_is_scoped_per_run(data_dir):
    fetch_data(n_samples=100, run_id="run-a", data_dir=data_dir)
    fetch_data(n_samples=100, run_id="run-b", data_dir=data_dir)
    for run_id in ("run-a", "run-b"):
        with open(run_paths(run_id, data_dir)["run_log"]) as f:
            log = json.load(f)
        assert len(log) == 1


def test_run_log_tracks_all_steps(data_dir, paths):
    fetch_data(n_samples=500, seed=1, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)
    evaluate_model(run_id="test-run", data_dir=data_dir)
    with open(paths["run_log"]) as f:
        log = json.load(f)
    steps = [entry["step"] for entry in log]
    assert "fetch_data" in steps
    assert "preprocess_data" in steps
    assert "train_model" in steps
    assert "evaluate_model" in steps


def test_train_saves_shared_drift_reference(data_dir, paths):
    fetch_data(n_samples=300, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)
    assert os.path.exists(paths["reference"])
    # The reference lives beside the runs/ tree, not inside one run.
    assert os.path.dirname(paths["reference"]) == paths["data_dir"]
    ref = pd.read_csv(paths["reference"])
    assert "churn" not in ref.columns


def test_check_drift_same_distribution_no_drift(data_dir, paths):
    fetch_data(n_samples=1000, seed=1, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)
    # Same distribution, new sample, arriving as a separate monitoring run.
    fetch_data(n_samples=1000, seed=2, run_id="monitor-run", data_dir=data_dir)
    report = check_drift(run_id="monitor-run", data_dir=data_dir)
    assert report["drift_detected"] is False
    assert os.path.exists(run_paths("monitor-run", data_dir)["drift_report"])


def test_check_drift_detects_shifted_data(data_dir, paths, tmp_path):
    fetch_data(n_samples=1000, seed=1, run_id="test-run", data_dir=data_dir)
    preprocess_data(run_id="test-run", data_dir=data_dir)
    train_model(run_id="test-run", data_dir=data_dir)

    shifted = pd.read_csv(paths["raw"])
    shifted["monthly_charges"] += 100
    shifted["tenure"] += 36
    shifted["support_calls"] += 5
    shifted["num_products"] += 2
    shifted_path = str(tmp_path / "shifted.csv")
    shifted.to_csv(shifted_path, index=False)

    report = check_drift(current_path=shifted_path, run_id="test-run", data_dir=data_dir)
    assert report["drift_detected"] is True
    assert "monthly_charges" in report["drifted_features"]


def test_check_drift_without_reference_skips(data_dir):
    fetch_data(n_samples=200, run_id="test-run", data_dir=data_dir)
    report = check_drift(run_id="test-run", data_dir=data_dir)
    assert report["drift_detected"] is False
    assert report["reason"] == "no_reference_data"


def test_run_pipeline(data_dir):
    report = run_pipeline(n_samples=300, seed=42, run_id="test-run", data_dir=data_dir)
    assert report["status"] == "pass"
    assert report["metrics"]["roc_auc"] >= 0.70


def test_run_pipeline_custom_params(data_dir):
    report = run_pipeline(
        n_samples=300,
        seed=42,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        run_id="test-run",
        data_dir=data_dir,
    )
    assert "metrics" in report
    assert "roc_auc" in report["metrics"]


def test_run_pipeline_clears_log(data_dir, paths):
    fetch_data(n_samples=100, run_id="test-run", data_dir=data_dir)
    run_pipeline(n_samples=500, seed=42, run_id="test-run", data_dir=data_dir)
    with open(paths["run_log"]) as f:
        log = json.load(f)
    steps = [e["step"] for e in log]
    assert steps.count("fetch_data") == 1


def _tree(root: str) -> set[str]:
    """Recursive listing of `root` as relative paths; empty if it does not exist."""
    if not os.path.isdir(root):
        return set()
    entries = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            entries.add(os.path.relpath(os.path.join(dirpath, name), root))
    return entries


def test_nothing_is_written_to_the_repo_data_dir(data_dir, paths):
    """Guards the isolation itself: a real run must leave ./data untouched."""
    repo_data_dir = os.path.abspath(DATA_DIR)
    existed_before = os.path.exists(repo_data_dir)
    before = _tree(repo_data_dir)

    run_pipeline(n_samples=300, seed=42, run_id="test-run", data_dir=data_dir)

    assert os.path.exists(repo_data_dir) == existed_before, (
        f"the run created {repo_data_dir}, which should not exist"
    )
    assert _tree(repo_data_dir) == before, (
        f"the run wrote into the repo data dir: {sorted(_tree(repo_data_dir) - before)}"
    )

    # ...and the artifacts really were produced, under tmp_path.
    for key in ("raw", "processed", "model", "report", "run_log", "reference"):
        assert os.path.exists(paths[key])
