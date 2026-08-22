"""
Churn prediction ML pipeline DAG.

Runs daily at midnight. Steps:
    1. fetch_data    - generate/pull raw data for this run
    2. preprocess    - scale features
    3. train         - train GradientBoosting, log to MLflow
    4. evaluate      - check quality gate (ROC-AUC >= 0.72)

Artifacts are namespaced by Airflow run id, so a run never reads or writes
another run's files.

On failure, the task is marked failed and `log_task_failure` writes a
structured JSON record (dag id, task id, run id, try number, exception) to
the Airflow task log. That is the whole of it: no email and no paging are
configured for this DAG.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import evaluate_model, fetch_data, preprocess_data, train_model  # noqa: E402

log = logging.getLogger("airflow.task")


def log_task_failure(context) -> None:
    """Emit one structured, greppable failure record to the task log."""
    ti = context.get("task_instance")
    record = {
        "event": "task_failure",
        "dag_id": getattr(ti, "dag_id", None),
        "task_id": getattr(ti, "task_id", None),
        "run_id": getattr(ti, "run_id", None),
        "try_number": getattr(ti, "try_number", None),
        "logical_date": str(context.get("logical_date") or context.get("execution_date")),
        "exception": repr(context.get("exception")),
    }
    log.error("PIPELINE_FAILURE %s", json.dumps(record, default=str))


default_args = {
    "owner": "yigit",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "on_failure_callback": log_task_failure,
}

with DAG(
    dag_id="churn_prediction_pipeline",
    description="Daily churn prediction training pipeline",
    schedule="0 0 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    # A hung run must not hold a LocalExecutor slot until someone notices.
    dagrun_timeout=timedelta(hours=1),
    tags=["ml", "churn", "production"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
        op_kwargs={"run_id": "{{ run_id }}", "logical_date": "{{ ds }}"},
        execution_timeout=timedelta(minutes=10),
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        op_kwargs={"run_id": "{{ run_id }}"},
        execution_timeout=timedelta(minutes=10),
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "run_id": "{{ run_id }}",
        },
        execution_timeout=timedelta(minutes=30),
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        # Stricter than src.pipeline's DEFAULT_MIN_ROC_AUC (0.70); see README.
        op_kwargs={"min_roc_auc": 0.72, "run_id": "{{ run_id }}"},
        execution_timeout=timedelta(minutes=5),
    )

    t1 >> t2 >> t3 >> t4
