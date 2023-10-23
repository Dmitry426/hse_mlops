import json
import logging
import pickle
from datetime import datetime, timedelta

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow.utils.dates import days_ago

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from services.postgres.connection import get_postgres_conn
from services.s3.connection import get_s3_hook, get_s3_resource

postgres_conn_name = "pg_connection"
s3_connection = "minio_s3"
bucket = "airflow"
data_path = "datasets/"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Dmitrii",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "catchup": False,
}

dag = DAG(
    "ml_ops_dag1",
    tags=["mlops"],
    catchup=False,
    start_date=days_ago(2),
    default_args=DEFAULT_ARGS,
    schedule_interval="@once",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def init() -> None:
    logger.info(" Pipline started ")


def get_housing_data() -> None:
    """Get housing data from bd"""
    conn = get_postgres_conn(postgres_conn_name)
    data = pd.read_sql_query("SELECT * FROM california_housing", conn)

    resource = get_s3_resource(s3_connection)
    pickl_dump_obj = pickle.dumps(data)

    path = data_path + "california_housing.pk1"

    resource.Object(bucket, path).put(Body=pickl_dump_obj)


def prepare_data() -> None:
    """Prepare data to train"""
    path = data_path + "california_housing.pk1"
    s3 = get_s3_hook(s3_connection)
    file = s3.download_file(key=path, bucket_name=bucket)
    data = pd.read_pickle(file)

    x, y = data[FEATURES], data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    x_train_fitted = scaler.fit_transform(x_train)

    x_test_fitted = scaler.transform(x_test)

    resource = get_s3_resource(s3_connection)

    for name, data in zip(
        ["x_train", "x_test", "y_train", "y_test"],
        [x_train_fitted, x_test_fitted, y_train, y_test],
    ):
        pickl_dump_obj = pickle.dumps(data)
        path = data_path + f"{name}.pk1"
        resource.Object(bucket, path).put(Body=pickl_dump_obj)

        logger.info("Data preparation finished")


def train_model():
    data = {}
    s3 = get_s3_hook(s3_connection)

    for name in ["x_train", "x_test", "y_train", "y_test"]:
        path = data_path + f"{name}.pk1"

        file = s3.download_file(key=path, bucket_name=bucket)
        data[name] = pd.read_pickle(file)

    model = RandomForestRegressor()
    model.fit(data["x_train"], data["y_train"])
    prediction = model.predict(data["x_test"])
    result = {
        "r2_score": r2_score(data["y_test"], prediction),
        "rmse": (mean_squared_error(data["y_test"], prediction) ** 0, 5),
        "mae": (median_absolute_error(data["y_test"], prediction) ** 0, 5),
    }

    date = datetime.now().strftime("%Y_%m_%d_&H")
    resource = get_s3_resource(s3_connection)
    json_obj = json.dumps(result)
    path = data_path + f"{date}.json"
    resource.Object(bucket, path).put(Body=json_obj)

    logger.info("Model train finished")


def save_results():
    logger.info("Success")


task_init = PythonOperator(
    task_id="Init", provide_context=True, python_callable=init, dag=dag
)

task_get_data = PythonOperator(
    task_id="get_housing_data",
    provide_context=True,
    python_callable=get_housing_data,
    dag=dag,
)
task_prepare_data = PythonOperator(
    task_id="prepare_data", provide_context=True, python_callable=prepare_data, dag=dag
)

task_train_model = PythonOperator(
    task_id="train_model", provide_context=True, python_callable=train_model, dag=dag
)

task_save_results = PythonOperator(
    task_id="log_result", provide_context=True, python_callable=save_results, dag=dag
)

task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
