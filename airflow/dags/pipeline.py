from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator,BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import datetime
import json
from pathlib import Path


