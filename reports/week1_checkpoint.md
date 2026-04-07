# Week 1 Checkpoint

This file captures the Week 1 project state after the first full anomaly-detection milestone.

## Checkpoint Summary

Week 1 is complete with the following pieces in place:

- EDA report committed to the repo
- Kafka telemetry producer implemented
- Kafka consumer path into TimescaleDB implemented
- batch CSV to TimescaleDB DAG working
- anomaly baseline comparison implemented
- LSTM autoencoder comparison implemented
- anomaly experiment artifacts saved to disk
- MLflow logging path added for anomaly comparison runs

## What is complete

### 1. EDA report

The Week 1 EDA report is available at:

- `reports/week1_eda_report.md`

It summarizes:

- dataset shape
- sensor variability
- constant columns
- degradation drift signals
- missing-value patterns
- stationarity checks
- FFT snapshots

### 2. Kafka to TimescaleDB pipeline

The streaming path now has both sides:

- producer: `src/stream_sensor_producer.py`
- consumer: `src/kafka_to_timescaledb_consumer.py`

The storage path is:

- `Kafka raw-sensor-data topic`
- consume and normalize telemetry messages
- insert into `telemetry.sensor_readings` in TimescaleDB

### 3. Batch pipeline

The Airflow batch DAG is:

- `dags/batch_csv_to_timescaledb.py`

It runs:

- CSV ingestion
- validation
- feature engineering
- TimescaleDB feature loading

### 4. Anomaly model comparison

Current anomaly methods implemented:

- `Isolation Forest`
- `Local Outlier Factor`
- `z-score`
- `MAD`
- `LSTM Autoencoder`

The current saved comparison shows:

- `MAD` and `z-score` are the strongest methods on the synthetic window-level setup
- the `LSTM Autoencoder` is strong and competitive
- the deep model is not automatically better than simpler robust baselines

### 5. MLflow logging

Anomaly comparison runs can now be logged to MLflow by running:

```bash
python -m src.anomaly_lstm_autoencoder --epochs 20 --sequence-length 30 --stride 5 --run-name week1_checkpoint --log-mlflow
```

This logs:

- run configuration
- model metrics
- best-model summary
- saved CSV and markdown artifacts

## Validation Status

Current validation command:

```bash
python -m pytest tests/test_anomaly_baseline.py tests/test_anomaly_lstm_autoencoder.py tests/test_kafka_to_timescaledb_consumer.py -q
```

Current result:

- `12 passed`

## Week 1 Outcome

By the end of Week 1, the project is no longer just a collection of notebooks and ideas.

It now has:

- a data understanding layer
- time-series storage
- streaming and batch ingestion paths
- engineered telemetry features
- classical anomaly baselines
- a deep sequence anomaly model
- reproducible experiment outputs
- a checkpoint-level EDA artifact
