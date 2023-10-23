import json
import logging
import pickle
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict

import backoff
import mlflow
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook

from airflow.utils.dates import days_ago
from boto3.resources.base import ServiceResource
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ParamValidationError,
)
from mlflow import MlflowException
from mlflow.models import infer_signature
from psycopg2 import OperationalError
from psycopg2._psycopg import connection

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

NAME = "_KachkinDmitrii10"

BUCKET = "lizvladi-mlops"

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

EXPERIMENT_NAME = NAME
DAG_ID = NAME
PARENT_RUN_NAME = "test"

postgres_conn_name = "pg_connection"
s3_connection = "minio_s3"

MODELS = dict(
    zip(
        ("rf", "lr", "hgb"),
        (RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()),
    )
)
DEFAULT_ARGS = {
    "owner": "Dmitrii",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "catchup": False,
}
dag = DAG(
    DAG_ID,
    tags=["mlops"],
    catchup=False,
    start_date=days_ago(2),
    default_args=DEFAULT_ARGS,
    schedule_interval="@once",
)


def give_up_connection() -> None:
    logger.error(f"Failed to connect to storage . Please check connect options or host")


@backoff.on_exception(
    backoff.expo,
    OperationalError,
    max_time=60,
    max_tries=3,
    on_giveup=give_up_connection,
)
def get_postgres_conn(postgres_conn) -> connection:
    """Postgres hook interface , in case you need changes code encapsulated  """
    hook = PostgresHook(postgres_conn_name)
    conn = hook.get_conn()
    return conn


@lru_cache(maxsize=None)
def get_s3_hook(s3_conn) -> S3Hook:
    """Get s3 hook object . Use cache to make singleton"""
    try:
        return S3Hook(
            s3_connection,
            transfer_config_args={
                "use_threads": False,
            },
        )
    except (ParamValidationError, ClientError) as error:
        logger.error("S3 credentials err")
        raise ValueError(
            "The AWS parameters you provided are incorrect: {}".format(error)
        ) from error


@backoff.on_exception(
    backoff.expo,
    (EndpointConnectionError, ConnectionError, ConnectTimeoutError),
    max_time=60,
    max_tries=3,
    on_giveup=give_up_connection,
)
def get_s3_resource(s3_conn) -> ServiceResource:
    """
    Interface to  get resource .
    I use Minio ,so if you need to apple some changes code encapsulated
    """
    try:
        s3 = get_s3_hook(s3_connection)
        session = s3.get_session(s3.conn_config.region_name)
        return session.resource("s3", endpoint_url=s3.conn_config.endpoint_url)
    except (ParamValidationError, ClientError) as error:
        logger.error("S3 credentials err")
        raise ValueError(
            "The AWS parameters you provided are incorrect: {}".format(error)
        ) from error


def init() -> Dict[str, Any]:
    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "start_timestamp": datetime.now().strftime("%Y%m%d %H:%M"),
    }

    existing_exp = mlflow.get_experiment_by_name(name=metrics["experiment_name"])
    if existing_exp:
        msg = "You trying to run already existing experiment , please rename it  "
        logger.error(msg)
        raise MlflowException(msg)

    if existing_exp and existing_exp.lifecycle_stage == "deleted":
        msg = (
            "You trying to run deleted experiment please rename your experiment "
            "or recover deleted"
        )
        logger.error(msg)
        raise MlflowException(msg)

    mlflow.create_experiment(
        name=metrics["experiment_name"],
        artifact_location=f"s3://{BUCKET}/{EXPERIMENT_NAME}",
    )

    experiment = mlflow.set_experiment(metrics["experiment_name"])
    metrics["experiment_id"] = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=metrics["experiment_id"],
        tags={"version": "v1", "priority": "P1"},
        description="Parent_run",
        run_name=PARENT_RUN_NAME,
    ) as parent_run:
        metrics["run_id"] = parent_run.info.run_id

    return metrics


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    """Get housing data from postgres and pickle in S3"""

    ti = kwargs["ti"]

    metrics = ti.xcom_pull(task_ids="init")

    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    conn = get_postgres_conn(postgres_conn_name)
    data = pd.read_sql_query("SELECT * FROM california_housing", conn)

    resource = get_s3_resource(s3_connection)
    pickl_dump_obj = pickle.dumps(data)

    path = f"{NAME}/datasets/california_housing.pkl"

    resource.Object(BUCKET, path).put(Body=pickl_dump_obj)

    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    """Prepare data to train and save result in s3"""
    ti = kwargs["ti"]

    metrics = ti.xcom_pull(task_ids="get_data_from_postgres")
    metrics["data_preparation_start"] = datetime.now().strftime("%Y_%m_%d_&H")

    s3 = get_s3_hook(s3_connection)
    file = s3.download_file(
        key=f"{NAME}/datasets/california_housing.pkl", bucket_name=BUCKET
    )

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
        path = NAME + f"/datasets/{name}.pk1"
        resource.Object(BUCKET, path).put(Body=pickl_dump_obj)

    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    return metrics


def train_mlflow_model(
    model: Any,
    name: str,
    x_train: np.array,
    x_test: np.array,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Train and log model metrics using mlflow"""

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    r2 = r2_score(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction)
    mae = median_absolute_error(y_test, prediction)

    # firstly save the model and then log_metric in case save is not possible

    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rsme", rmse)
    mlflow.log_metric("mae", mae)

    signature = infer_signature(x_test, prediction)
    model_info = mlflow.sklearn.log_model(model, name, signature=signature)

    mlflow.evaluate(
        model_info.model_uri,
        data=x_test,
        targets=y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )


#
def train_model(model_name: str, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]

    metrics = ti.xcom_pull(task_ids="prepare_data")
    metrics[f"train_start_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")
    data = {}
    s3 = get_s3_hook(s3_connection)

    model = MODELS[model_name]

    for name in ["x_train", "x_test", "y_train", "y_test"]:
        file = s3.download_file(key=NAME + f"/datasets/{name}.pk1", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    mlflow.set_experiment(metrics["experiment_name"])

    with mlflow.start_run(run_id=metrics["run_id"]):
        with mlflow.start_run(run_name=model_name, nested=True):
            train_mlflow_model(
                model=model,
                name=model_name,
                x_train=data["x_train"],
                x_test=data["x_test"],
                y_train=data["y_train"],
                y_test=data["y_test"],
            )

    metrics[f"train_end_{model_name}"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics


def save_results(**kwargs) -> None:
    ti = kwargs["ti"]
    result = []
    for model in MODELS.keys():
        metrics = ti.xcom_pull(task_ids=f"train_{model}")
        logger.info(metrics)
        metrics[f"end_time"] = datetime.now().strftime("%Y%m%d %H:%M")
        result.append(metrics)
        logger.info(f"Model {model} train finished")

    end_time = datetime.now().strftime("%Y%m%d %H:%M")
    path = f"{NAME}/results/{end_time}.json"
    resource = get_s3_resource(s3_connection)
    json_obj = json.dumps(result)

    resource.Object(BUCKET, path).put(Body=json_obj)


with dag:
    task_init = PythonOperator(
        task_id="init", provide_context=True, python_callable=init, dag=dag
    )

    task_get_data = PythonOperator(
        task_id="get_data_from_postgres",
        provide_context=True,
        python_callable=get_data_from_postgres,
        dag=dag,
    )
    task_prepare_data = PythonOperator(
        task_id="prepare_data",
        provide_context=True,
        python_callable=prepare_data,
        dag=dag,
    )

    tasks = []
    for m_name in MODELS.keys():
        task = PythonOperator(
            task_id=f"train_{m_name}",
            python_callable=train_model,
            op_args=[m_name],
            dag=dag,
        )
        tasks.append(task)

    task_save_results = PythonOperator(
        task_id="save_results",
        provide_context=True,
        python_callable=save_results,
        dag=dag,
    )

task_init >> task_get_data >> task_prepare_data >> tasks >> task_save_results
