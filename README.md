# Predictive Maintenance and RUL Forecasting Platform

This repository is for building a predictive maintenance project around Remaining Useful Life (RUL) forecasting, anomaly detection, and maintenance decision support.

The project is based on the LogicVeda capstone brief, but the repo is being developed as a learning-driven engineering project, not as a day-by-day checklist dump.

## Project goal

Build an end-to-end system that can:
- analyze equipment sensor data
- detect abnormal behavior
- estimate remaining useful life
- support maintenance planning
- evolve toward a deployable application

## How this repo is meant to be used

This repo has two jobs:
- project implementation
- learning record

Implementation lives in code and notebooks.
Learning lives in the journal, where concepts, definitions, decisions, library usage, and mistakes are documented as the project grows.

## Repository structure

- `src/` - reusable Python code for data loading, EDA helpers, feature engineering, modeling, and utilities
- `notebooks/` - exploratory work and analysis notebooks
- `Data/` - source datasets used for the project
- `app/` - future app or dashboard code
- `deploy/` - future deployment-related files
- `tests/` - test files as the project becomes more structured
- `journal/` - learning notes, definitions, why a method was used, library explanations, and mistakes

## Current focus

Right now the project is still in the early data-understanding stage.
The main dataset is NASA CMAPSS, which is the core dataset for learning the RUL problem properly.

## Day 4 pipeline slice

The repo now also includes a first batch feature pipeline for telemetry-style CSV data:
- `dags/batch_csv_to_timescaledb.py` orchestrates batch CSV ingestion, validation, feature engineering, and TimescaleDB loading
- `src/batch_pipeline.py` stages raw CSV batches and writes engineered feature CSVs
- `src/batch_validation.py` runs batch validation and uses Great Expectations when it is installed
- `src/feature_engineering.py` builds rolling 1h/8h/24h statistics, lag-1 to lag-12 values, FFT top-5 amplitudes, and cross-sensor ratios
- `src/ingest_feature_timescaledb.py` loads engineered features into the `telemetry.sensor_features` hypertable

The feature store table is created in:
- `db/init/002_sensor_features.sql`

If you want the orchestration dependencies separately from the lighter project dependencies, install:

```bash
pip install -r requirements-orchestration.txt
```

## Airflow runtime

The project now includes a Dockerized Airflow runtime for the Day 4 batch DAG.

Files involved:
- `dags/batch_csv_to_timescaledb.py`
- `deploy/airflow/Dockerfile`
- `docker-compose.yml`

Bring it up with:

```bash
docker compose up -d timescaledb airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

Then open:
- `http://localhost:8080`

Default Airflow login:
- username: `admin`
- password: `admin`

The DAG you can run is:
- `batch_csv_to_timescaledb`

What it does:
1. Reads the batch CSV from `Data/batch/sensor_batch.csv`
2. Normalizes and stages it into `Data/batch/staging/normalized_sensor_batch.csv`
3. Validates the staged data with Great Expectations-compatible checks
4. Engineers rolling, lag, FFT, and ratio features into `Data/batch/staging/engineered_sensor_features.csv`
5. Loads the engineered rows into `telemetry.sensor_features` in TimescaleDB

You can also trigger it from the CLI:

```bash
docker exec project1-airflow-scheduler airflow dags trigger batch_csv_to_timescaledb
```

## Working style

- keep notebooks for exploration
- keep reusable logic in `src/`
- write down reasoning, not just results
- separate learning value from production realism
- avoid adding infrastructure before the core ML problem is understood

## Learning journal

Use the journal as a technical notebook for yourself, not as polished documentation.

Suggested things to capture:
- definitions of new terms
- what a dataset or feature actually means
- why a method was chosen
- what each library is doing in the project
- what can go wrong
- what still feels unclear

Start here:
- `journal/learning_journal.md`

## Notes

This README should stay high level.
Daily execution details, detailed notes, and learning reflections belong in the journal or project-specific notebooks, not here.
