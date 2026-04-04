ARG AIRFLOW_VERSION=2.10.5
FROM apache/airflow:${AIRFLOW_VERSION}-python3.11

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11

USER airflow

RUN pip install --no-cache-dir \
    "pandas" \
    "numpy" \
    "psycopg[binary]" \
    "great-expectations>=0.18,<1" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
