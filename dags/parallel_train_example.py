import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow.utils.dates import days_ago
from pydantic.v1 import BaseModel

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from services.postgres.connection import get_postgres_conn
from services.s3.connection import get_s3_hook, get_s3_resource


class Connections(BaseModel):
    postgres_conn_name: str = "pg_connection"
    s3_connection: str = "minio_s3"
    bucket: str = "airflow"


class DagDefaultArgs(BaseModel):
    owner: str = "Dmitrii"
    retries: int = 1
    depends_on_past: bool = False
    retry_delay: timedelta = timedelta(minutes=1)
    start_date: datetime = days_ago(2)


class DagSettings(BaseModel):
    dag_id: str = "parallel_train"
    catchup: bool = False
    start_date: datetime = days_ago(2)
    schedule_interval: str = "@once"


class ModelSettings(BaseModel):
    features: List = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target: str = "MedHouseVal"
    models: Dict = dict(
        zip(
            ("rf", "lr", "hgb"),
            (
                RandomForestRegressor(),
                LinearRegression(),
                HistGradientBoostingRegressor(),
            ),
        )
    )


class ModelTrainYamlSettings(BaseModel):
    dag_settings: DagSettings = DagSettings()
    default_args: DagDefaultArgs = DagDefaultArgs()
    connections: Connections = Connections()
    model: ModelSettings = ModelSettings()


settings = ModelTrainYamlSettings()

dag = DAG(
    dag_id=settings.dag_settings.dag_id,
    tags=["mlops"],
    catchup=settings.dag_settings.catchup,
    start_date=settings.dag_settings.start_date,
    default_args=settings.default_args.dict(),
    schedule_interval=settings.dag_settings.schedule_interval,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def init() -> Dict[str, Any]:
    metrics = {"timestamp": datetime.now().strftime("%Y%m%d %H:%M")}
    return metrics


def get_housing_data(**kwargs) -> Dict[str, Any]:
    """Get housing data from postgres and update s3"""

    ti = kwargs["ti"]

    metrics = ti.xcom_pull(task_ids="init")

    conn = get_postgres_conn(postgres_conn_name=settings.connections.postgres_conn_name)
    data = pd.read_sql_query("SELECT * FROM california_housing", conn)

    resource = get_s3_resource(s3_connection=settings.connections.s3_connection)
    pickl_dump_obj = pickle.dumps(data)

    path = f"{settings.dag_settings.dag_id}/california_housing.pk1"

    resource.Object(settings.connections.bucket, path).put(Body=pickl_dump_obj)

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    """Prepare data to train"""
    ti = kwargs["ti"]

    metrics = ti.xcom_pull(task_ids="get_housing_data")

    path = f"{settings.dag_settings.dag_id}/california_housing.pk1"
    s3 = get_s3_hook(s3_connection=settings.connections.s3_connection)
    file = s3.download_file(key=path, bucket_name=settings.connections.bucket)
    data = pd.read_pickle(file)

    x, y = data[settings.model.features], data[settings.model.target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_fitted = scaler.fit_transform(x_train)
    x_test_fitted = scaler.transform(x_test)

    resource = get_s3_resource(s3_connection=settings.connections.s3_connection)

    for name, data in zip(
        ["x_train", "x_test", "y_train", "y_test"],
        [x_train_fitted, x_test_fitted, y_train, y_test],
    ):
        pickl_dump_obj = pickle.dumps(data)
        path = f"{settings.dag_settings.dag_id}/{name}.pk1"
        resource.Object(settings.connections.bucket, path).put(Body=pickl_dump_obj)

        logger.info("Data preparation finished")

    return metrics


def train_model(model_name: str, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]

    metrics = ti.xcom_pull(task_ids="prepare_data")
    metrics["train_started"] = datetime.now().strftime("%Y%m%d %H:%M")

    hook = get_s3_hook(s3_connection=settings.connections.s3_connection)

    data = {}

    for train in ["x_train", "x_test", "y_train", "y_test"]:
        file = hook.download_file(
            key=f"{settings.dag_settings.dag_id}/{train}.pk1",
            bucket_name=settings.connections.bucket,
        )
        data[train] = pd.read_pickle(file)

    model = settings.model.models[model_name]
    metrics["model"] = model_name
    model.fit(data["x_train"], data["y_train"])
    prediction = model.predict(data["x_test"])
    metrics["train_end"] = datetime.now().strftime("%Y_%m_%d_&H")

    train_result = {
        "r2_score": r2_score(data["y_test"], prediction),
        "rmse": (mean_squared_error(data["y_test"], prediction) ** 0, 5),
        "mae": (median_absolute_error(data["y_test"], prediction) ** 0, 5),
    }

    return {**metrics, **train_result}


def save_results(**kwargs) -> None:
    ti = kwargs["ti"]

    result = []
    for model in settings.model.models.keys():
        metrics = ti.xcom_pull(task_ids=f"train_{model}")
        metrics["end"] = datetime.now().strftime("%Y_%m_%d_&H")

        result.append(metrics)

    resource = get_s3_resource(s3_connection=settings.connections.s3_connection)

    json_obj = json.dumps(result)

    end_time = datetime.now().strftime("%Y%m%d %H:%M")
    path = f"{settings.dag_settings.dag_id}/results/{end_time}.json"

    resource.Object(settings.connections.bucket, path).put(Body=json_obj)

    logger.info("Model train finished")


with dag:
    task_init = PythonOperator(
        task_id="init", provide_context=True, python_callable=init, dag=dag
    )

    task_get_data = PythonOperator(
        task_id="get_housing_data",
        provide_context=True,
        python_callable=get_housing_data,
        dag=dag,
    )
    task_prepare_data = PythonOperator(
        task_id="prepare_data",
        provide_context=True,
        python_callable=prepare_data,
        dag=dag,
    )

    tasks = []
    for name in settings.model.models.keys():
        task = PythonOperator(
            task_id=f"train_{name}",
            python_callable=train_model,
            op_args=[name],
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
