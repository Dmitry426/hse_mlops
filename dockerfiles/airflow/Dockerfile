FROM apache/airflow:latest
COPY ./requirements.txt /
RUN pip install -U jupyter-core --user
RUN pip install -U jupyter --user
RUN pip install --no-cache-dir -r /requirements.txt
RUN chmod -R 775 /home/airflow/.local/share/jupyter
