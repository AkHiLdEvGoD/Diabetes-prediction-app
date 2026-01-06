from datetime import datetime
import time

from airflow.sdk import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator


def extract_data():
    print("Extracting data...")
    time.sleep(2)
    print("Data extracted successfully")


def process_data():
    print("Processing data...")
    time.sleep(3)
    print("Data processed successfully")


def save_data():
    print("Saving data...")
    time.sleep(1)
    print("Data saved successfully")


with DAG(
    dag_id="demo_python_operator",
    start_date=datetime(2024, 1, 1),
    schedule=None,          # manual trigger
    catchup=False,
    tags=["demo", "python-operator"],
) as dag:

    start = EmptyOperator(
        task_id="start"
    )

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    process = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
    )

    save = PythonOperator(
        task_id="save_data",
        python_callable=save_data,
    )

    end = EmptyOperator(
        task_id="end"
    )

    # Task dependencies
    start >> extract >> process >> save >> end
