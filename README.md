# Airflow ML Pipeline

![CI](https://github.com/yigitliman/airflow-ml-pipeline/actions/workflows/ci.yml/badge.svg)

An end-to-end machine learning pipeline orchestrated with Apache Airflow. Runs on a daily schedule and covers data fetching, preprocessing, model training, and quality-gated evaluation. Experiment tracking via MLflow.

## Pipeline Steps

```
fetch_data -> preprocess_data -> train_model -> evaluate_model
```

Each step is a separate Airflow task. If the model fails the quality gate, the DAG is marked as failed and a structured failure record (dag id, task id, run id, try number, exception) is written to the Airflow task log. No email or paging is configured.

There are two gate values on purpose. `src/pipeline.py` ships `DEFAULT_MIN_ROC_AUC = 0.70` as the library floor; the DAG overrides it with `min_roc_auc=0.72` so a scheduled retrain holds a stricter line than a manual run. Each scheduled run trains on a different synthetic batch (the seed comes from the logical date), so ROC-AUC moves between runs and an unlucky batch will fail the DAG.

## Architecture

Two DAGs work together: a daily training pipeline and a drift monitor that
triggers retraining when incoming data no longer matches the training data.

```mermaid
flowchart LR
    subgraph training [churn_prediction_pipeline, daily]
        F[fetch_data] --> P[preprocess_data] --> TR[train_model] --> E[evaluate_model]
        TR -- params and metrics --> M[MLflow]
        E -- "ROC-AUC >= 0.72" --> OK[report.json]
        E -- "ROC-AUC < 0.72" --> FAIL[DAG marked failed]
    end
    subgraph monitoring [drift_monitoring, every 6 hours]
        FB[fetch_fresh_batch] --> CD{check_drift KS test}
        CD -- drift --> TRIG[trigger_retraining]
        CD -- no drift --> NOOP[no_drift]
    end
    TR -- reference.csv --> CD
    TRIG -. triggers .-> F
```

## Results

One full pipeline run on the seeded synthetic dataset (2,000 rows, 400 held out, `random_state=42`) with the default hyperparameters. Scheduled runs now seed the data generator from the logical date, so reproduce this exact run with an explicit seed:

```python
run_pipeline(n_samples=2000, seed=42)
```

| Metric | Value |
|---|---|
| ROC-AUC | 0.755 |
| Accuracy | 0.788 |
| F1 | 0.206 |
| Precision / Recall | 0.423 / 0.136 |

**Why the quality gate is on ROC-AUC and not accuracy.** Only 20% of the held-out customers actually churn, so a model that always answers "no churn" scores 0.798 accuracy, *better* than this model's 0.788. Accuracy on imbalanced churn data mostly measures the class balance, which is why gating on it would let a useless model through. ROC-AUC is threshold-free and reads 0.755 here, so the model does rank churners above non-churners; the low recall says the default 0.5 cutoff is simply the wrong operating point for a 20% positive rate, not that the signal is missing.

The data is generated from a known churn-probability function with binomial noise, so these numbers are what a well-behaved model on genuinely noisy data looks like, not a benchmark against real customer data. `fetch_data` is a synthetic generator standing in for a real source system: it fabricates each batch in-process and seeds itself from the run's logical date, so each scheduled run gets different rows.

## Run Artifacts

Both DAGs call the same pipeline steps, so per-run outputs are namespaced by
Airflow run id and never collide:

```
data/reference.csv       shared, long-lived drift baseline (written by train_model)
data/runs/<run_id>/      raw.csv, processed.csv, model.joblib, report.json,
                         run_log.json, drift_report.json
```

The drift monitor runs every 6 hours and the training DAG runs daily, so
without this they would have raced on a single `data/raw.csv`. The run log is
scoped per run for the same reason, which also stops it growing without bound.

## Drift Detection

The training step snapshots its raw features as `reference.csv`. The
drift monitor compares each new data batch against that snapshot with a
Kolmogorov-Smirnov test per feature. When at least half the features
drift (p < 0.05), the monitor triggers the training DAG.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Orchestration | Apache Airflow 2.9 |
| Model | GradientBoosting (scikit-learn) |
| Experiment Tracking | MLflow |
| Database | PostgreSQL (Airflow metadata) |
| Containerization | Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |

## Quick Start

**Install dependencies (for running tests locally)**

```bash
pip install -r requirements.txt
pytest tests/ -v
```

**Install Airflow (for running the full DAG locally)**

```bash
pip install apache-airflow==2.9.2
```

**Run the full stack with Docker Compose**

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |

Default Airflow credentials: `airflow / airflow`

The `churn_prediction_pipeline` DAG will appear in the UI. Trigger it manually or wait for the daily schedule.

## DAG Structure

```python
t1 = fetch_data        # generate raw customer data
t2 = preprocess_data   # scale features with StandardScaler
t3 = train_model       # train GradientBoosting, log to MLflow
t4 = evaluate_model    # check ROC-AUC >= 0.72 (DAG override), write report.json

t1 >> t2 >> t3 >> t4
```

## Running Tests

Pipeline steps are tested independently without Airflow.

```bash
pytest tests/ -v
```

## CI/CD

Every push to `main` runs the full pipeline test suite with coverage, and a separate job installs Airflow 2.9.2 and loads the DagBag to assert both DAGs import without errors, so a broken DAG fails CI instead of surfacing on deploy.

## License

Released under the MIT License. See [LICENSE](LICENSE).
