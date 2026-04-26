# Predictive Maintenance Platform

An end-to-end predictive maintenance project built around sensor telemetry, time-series storage, feature engineering, anomaly detection, and sequence modeling.

This repository brings together:

- industrial-style telemetry pipelines
- batch and streaming data flow
- time-series database design
- classical anomaly detection baselines
- deep learning with an LSTM autoencoder

## Why This Project Exists

Predictive maintenance problems are rarely just "train a model on a CSV."

Real maintenance systems usually involve:

- sensor streams that arrive over time
- telemetry that needs validation before it can be trusted
- time-aware storage and querying
- feature engineering across rolling windows
- anomaly detection before full failure labels are available

This project is built around that reality.

## Highlights

- Uses `NASA CMAPSS` as the main sequential maintenance dataset
- Supports `synthetic telemetry generation` for controlled anomaly experiments
- Stores telemetry in `TimescaleDB` using time-series structure
- Simulates real-time flow with `Kafka`
- Orchestrates batch ingestion with `Airflow`
- Engineers rolling, lag, FFT, and ratio-based sensor features
- Compares anomaly methods from simple statistical scoring to deep sequence models
- Saves experiment artifacts for reproducible review

## Results Snapshot

Current anomaly detection results on the synthetic window-level comparison:

| Method | ROC-AUC | PR-AUC | Summary |
| --- | --- | --- | --- |
| `MAD` | `0.9416` | `0.9400` | Strongest current method in this setup |
| `z-score` | `0.9407` | `0.9399` | Nearly tied with MAD and highly interpretable |
| `LSTM Autoencoder` | `0.9334` | `0.9285` | Strong temporal model with reconstruction-error scoring |
| `Local Outlier Factor` | `0.9183` | `0.9213` | Strong local-density baseline |
| `Isolation Forest` | `0.7594` | `0.6728` | Weakest of the current comparison set |

Main takeaway:
the LSTM autoencoder is competitive and learns useful sequence structure, but the simpler robust statistical methods are still slightly better on the current synthetic anomaly design.

## System View

```text
CMAPSS / Kaggle / Synthetic telemetry
                |
                +--> Kafka producer --> raw-sensor-data --> cleaned-features --> anomalies-flagged
                |
                +--> Batch CSV ingestion --> validation --> feature engineering --> TimescaleDB
                                                                    |
                                                                    +--> anomaly baselines
                                                                    +--> LSTM autoencoder
                                                                    +--> saved experiment artifacts
```

## Datasets

### NASA CMAPSS turbofan dataset

Used as the main sequential dataset for predictive maintenance and time-aware modeling.

### Kaggle predictive maintenance datasets

Used as supporting reference datasets for telemetry structure, maintenance-style sensor data, and comparison context.

### Synthetic telemetry

Used for controlled anomaly experiments where clean anomaly labels are needed for evaluation.

## Technology Stack

### Data and storage

- `PostgreSQL + TimescaleDB`
- `Kafka`
- `Airflow`
- `MLflow`

### Data processing and validation

- `pandas`
- `numpy`
- `Great Expectations`

### Machine learning and analysis

- `scikit-learn`
- `PyOD`
- `PyTorch`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`

## Pipeline Components

### Time-series storage

The telemetry schema supports fields such as:

- `timestamp`
- `equipment_id`
- `run_id`
- sensor readings
- `failure_label`

TimescaleDB is used so the data can be queried and aggregated in a time-aware way rather than treated like generic tabular storage.

### Streaming simulation

Kafka topics currently include:

- `raw-sensor-data`
- `cleaned-features`
- `anomalies-flagged`

The Python producer simulates mostly normal operation and injects anomalies at controlled intervals so downstream detection can be evaluated.

### Kafka to TimescaleDB ingestion

The stream-to-storage path now includes:

- `src/stream_sensor_producer.py`
- `src/kafka_to_timescaledb_consumer.py`

This makes the Week 1 Kafka to TimescaleDB checkpoint concrete rather than only conceptual.

### Batch feature pipeline

The batch pipeline supports:

- CSV ingestion
- validation checks
- staged normalization
- feature engineering
- loading engineered features into TimescaleDB

The Airflow DAG for this flow is:

- `batch_csv_to_timescaledb`

## Feature Engineering

The current telemetry feature pipeline builds:

- rolling `1h`, `8h`, and `24h` mean, std, min, and max
- `lag-1` to `lag-12`
- FFT top frequency amplitudes
- cross-sensor ratios

These features are designed to capture:

- local state
- volatility
- temporal memory
- frequency behavior
- relationships between sensors

## Anomaly Detection Methods

The anomaly track includes both baseline and sequence-based methods.

### Classical and statistical methods

- `Isolation Forest`
- `Local Outlier Factor`
- `z-score`
- `MAD` (Median Absolute Deviation)

These methods are useful because they are fast, interpretable, and provide strong baselines before moving into more complex models.

### Deep learning method

- `LSTM Autoencoder`

The LSTM autoencoder is trained on normal sequence windows and uses reconstruction error as the anomaly score. Thresholding is based on the upper quantile of train reconstruction errors.

## Repository Structure

```text
Project-1/
|-- src/            # reusable Python modules
|-- notebooks/      # analysis and experiment review notebooks
|-- dags/           # Airflow DAG definitions
|-- db/             # database initialization scripts
|-- Data/           # datasets, staged files, and experiment artifacts
|-- reports/        # generated EDA and checkpoint reports
|-- tests/          # pipeline and anomaly detection tests
|-- deploy/         # runtime and deployment helpers
|-- airflow/        # local Airflow runtime assets
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For orchestration-related dependencies:

```bash
pip install -r requirements-orchestration.txt
```

### 2. Run the baseline anomaly comparison

```bash
python -m src.anomaly_baseline
```

This runs:

- Isolation Forest
- Local Outlier Factor
- z-score
- MAD

### 3. Run the LSTM autoencoder comparison

```bash
python -m src.anomaly_lstm_autoencoder --epochs 20 --sequence-length 30 --stride 5 --run-name day6_final
```

This will:

- train the LSTM autoencoder
- score anomaly windows using reconstruction error
- compare it against the simpler methods
- save metrics and artifacts under `Data/experiments/anomaly_day6/`

### 4. Log the anomaly comparison to MLflow

```bash
python -m src.anomaly_lstm_autoencoder --epochs 20 --sequence-length 30 --stride 5 --run-name week1_checkpoint --log-mlflow
```

This logs:

- run parameters
- model metrics
- best-model summary
- saved artifacts

### 5. Run tests

```bash
python -m pytest tests/test_anomaly_baseline.py tests/test_anomaly_lstm_autoencoder.py tests/test_kafka_to_timescaledb_consumer.py -q
```

### 6. Run Day 8 sequence preparation

For continuous RUL windows:

```bash
python -m src.run_sequence_preparation --fd FD001 --target-mode rul --window-size 30 --stride 1
```

For binary failure forecasting over the next 30 cycles:

```bash
python -m src.run_sequence_preparation --fd FD001 --target-mode failure_in_next_window --window-size 30 --prediction-horizon 30
```

This prints:

- train, validation, and test window counts
- target summary for the selected task
- one batch shape from each loader
- a metadata preview showing how targets are aligned to the end of each window

### 7. Run Day 9 supervised sequence training with Optuna

For RUL regression with bidirectional LSTM + attention, Prophet trend/seasonality blending, and Optuna tuning:

```bash
python -m src.run_sequence_training --fd FD001 --target-mode rul --max-epochs 15 --optuna-trials 10
```

For binary failure forecasting over the next 30 cycles:

```bash
python -m src.run_sequence_training --fd FD001 --target-mode failure_in_next_window --prediction-horizon 30 --max-epochs 15 --optuna-trials 10
```

This workflow:

- trains a `PyTorch Lightning` bidirectional LSTM with temporal attention
- fits a `Prophet` baseline on the cycle-level RUL trend with seasonality decomposition
- blends the final RUL prediction as `0.7 * LSTM + 0.3 * Prophet` by default
- uses `Optuna` to tune learning rate, hidden size, window length, dropout, and layer count
- evaluates the best checkpoint on the CMAPSS test split
- saves metrics, predictions, Optuna trial history, and a markdown summary under `Data/experiments/day9_sequence_training/`

To log a comparable forecasting variant to MLflow:

```bash
python -m src.run_sequence_training --fd FD001 --target-mode rul --max-epochs 5 --optuna-trials 0 --disable-prophet-ensemble --log-mlflow --variant-name lstm_baseline
```

### 8. Run the maintenance scheduler and sensitivity analysis

To convert the latest saved forecast into a maintenance schedule:

```bash
python -m src.run_week2_checkpoint --predictions-csv Data/experiments/day9_sequence_training/fd001_live_check/test_predictions.csv --run-name week2_smoke
```

This workflow:

- builds a realistic maintenance cost matrix with downtime, repair, and labor costs
- solves a `PuLP` mixed-integer schedule with technician, parts, and downtime SLA constraints
- runs cost-parameter sensitivity analysis to measure schedule robustness
- saves the optimal schedule as JSON/CSV plus a cost-vs-risk visualization under `Data/experiments/week2_checkpoint/`

### 9. Launch the Streamlit dashboard skeleton

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes starter pages for:

- `Overview`
- `Equipment Detail`
- `Alerts Configuration`
- `Reports`

## Streamlit Deployment

The Streamlit application entrypoint is:

- `app/streamlit_app.py`

Verified public/static project report site:

- `https://raw.githack.com/Manireddy69/predictive-maintenance-rul-forecasting-platform/gh-pages/index.html`

GitHub Pages canonical URL, after enabling Pages source as `gh-pages / root` in repository settings:

- `https://manireddy69.github.io/predictive-maintenance-rul-forecasting-platform/`

Deployment link tracker:

- `reports/deployment_links.md`

### Local launch

```bash
streamlit run app/streamlit_app.py
```

### Docker deployment

Build the image from the project root:

```bash
docker build -t predictive-maintenance-rul-dashboard .
```

Run the container and expose the dashboard on port `8501`:

```bash
docker run --rm -p 8501:8501 predictive-maintenance-rul-dashboard
```

Then open:

- `http://localhost:8501`

### Streamlit Community Cloud

To publish this dashboard on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app in the Streamlit dashboard.
3. Select this repository and set the main file path to `app/streamlit_app.py`.
4. Deploy using the repository root `requirements.txt`.

The repository also includes:

- `.streamlit/config.toml` for Streamlit server settings and theming
- `deploy/streamlit/Dockerfile` for container-based deployment

## Airflow Runtime

To start the local Airflow and TimescaleDB stack:

```bash
docker compose up -d timescaledb airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

Then open:

- `http://localhost:8080`

Default credentials:

- username: `admin`
- password: `admin`

## Experiment Artifacts

LSTM comparison runs save outputs under:

- `Data/experiments/anomaly_day6/`

Each saved run includes:

- `comparison_metrics.csv`
- `window_scores.csv`
- `row_level_scores.csv`
- `training_history.csv`
- `run_metadata.json`
- `day_06_summary.md`

This makes model runs reviewable without retraining every time.

## Notebooks

Useful notebooks in the project:

- `01_cmapss_data_understanding.ipynb`
  Dataset understanding and early analysis

- `02_anomaly_baseline_visualization.ipynb`
  Visualization of the baseline anomaly experiment

- `03_lstm_autoencoder_comparison.ipynb`
  Review notebook for saved LSTM comparison artifacts

## Reports

Generated project reports currently include:

- `reports/week1_eda_report.md`
- `reports/week1_checkpoint.md`
- `reports/final_project_report_draft.md`

Final submission helpers:

- `journal/demo_video_script.md`
- `journal/screenshot_checklist.md`
- `journal/submission_checklist.md`
- `reports/compliance_report.md`

Verification evidence commands:

```bash
python -m src.evaluate_anomaly
python -m src.evaluate_rul
python -m src.benchmark_latency --iterations 3 --max-tasks 12
python -m src.render_report_html
```

## Current Project State

The repository currently has:

- a working time-series storage setup
- batch and streaming telemetry components
- engineered telemetry features
- anomaly detection baselines
- an LSTM autoencoder comparison workflow
- reproducible saved experiment outputs

The result is a predictive maintenance codebase that already has a real anomaly-detection pipeline, not just model code in isolation.
