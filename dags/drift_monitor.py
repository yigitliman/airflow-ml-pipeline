"""
Drift monitoring DAG.

Runs every 6 hours. Fetches a fresh batch of data, compares it against
the reference snapshot saved at training time, and triggers the training
DAG when drift is detected. This closes the monitor -> decide -> retrain loop.

    fetch_fresh_batch -> check_drift -+-> trigger_retraining (drift)
                                      +-> no_drift           (no drift)

The fresh batch is written under this run's own artifact directory, so the
monitor can never overwrite the input of an in-flight training run.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import check_drift, fetch_data  # noqa: E402

default_args = {
    "owner": "yigit",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def fetch_fresh_batch(run_id: str) -> None:
    # The seed is derived from the run id, so every 6-hourly run simulates a
    # different batch of incoming data while staying reproducible on retry.
    fetch_data(n_samples=2000, run_id=run_id)


def decide_on_drift(run_id: str) -> str:
    report = check_drift(run_id=run_id)
    return "trigger_retraining" if report["drift_detected"] else "no_drift"


with DAG(
    dag_id="drift_monitoring",
    description="Checks for data drift and triggers retraining when needed",
    schedule="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    dagrun_timeout=timedelta(minutes=30),
    tags=["ml", "monitoring", "drift"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_fresh_batch",
        python_callable=fetch_fresh_batch,
        op_kwargs={"run_id": "{{ run_id }}"},
        execution_timeout=timedelta(minutes=10),
    )

    t2 = BranchPythonOperator(
        task_id="check_drift",
        python_callable=decide_on_drift,
        op_kwargs={"run_id": "{{ run_id }}"},
        execution_timeout=timedelta(minutes=10),
    )

    trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="churn_prediction_pipeline",
        wait_for_completion=False,
    )

    no_drift = EmptyOperator(task_id="no_drift")

    t1 >> t2 >> [trigger, no_drift]
