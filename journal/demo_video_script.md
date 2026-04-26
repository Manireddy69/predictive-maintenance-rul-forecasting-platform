# Demo Video Script

Target length: 5 to 7 minutes.

## 0:00 - 0:30 Opening

Introduce the project:

"This project is a predictive maintenance and remaining useful life forecasting platform. The goal is to move from reactive equipment breakdowns to earlier anomaly detection, RUL forecasting, and cost-aware maintenance scheduling."

Show:

- project title
- repository root
- README overview

## 0:30 - 1:20 Data and Problem Setup

Explain:

- NASA CMAPSS is the main dataset
- each engine unit is a time sequence
- train units run until failure
- test units stop before failure
- RUL must be derived carefully to avoid target mistakes

Show:

- `Data/CMaps`
- `reports/week1_eda_report.md`
- EDA dataset summary and sensor findings

## 1:20 - 2:20 Ingestion and Feature Pipeline

Explain:

- the project includes both batch and streaming-style paths
- Kafka simulates telemetry streams
- Airflow represents scheduled batch ingestion
- TimescaleDB stores time-series readings and engineered features

Show:

- `src/stream_sensor_producer.py`
- `src/kafka_to_timescaledb_consumer.py`
- `dags/batch_csv_to_timescaledb.py`
- `src/feature_engineering.py`
- `db/init`

## 2:20 - 3:20 Anomaly Detection

Explain:

- anomaly detection was treated separately from RUL forecasting
- simple robust methods were compared against an LSTM autoencoder
- the best current anomaly result is MAD, not the deep model
- this is a strength because it shows evidence-based model selection

Show:

- `reports/week1_checkpoint.md`
- `Data/experiments/anomaly_day6/day6_final/comparison_metrics.csv`
- notebook `03_lstm_autoencoder_comparison.ipynb`

Mention:

- MAD ROC-AUC about `0.9416`
- LSTM autoencoder ROC-AUC about `0.9334`

## 3:20 - 4:20 RUL Forecasting

Explain:

- sequence windows are created from unit histories
- the model is a bidirectional LSTM with attention
- Optuna is used for tuning
- outputs are saved as predictions and metrics

Show:

- `src/sequence_data.py`
- `src/sequence_attention_model.py`
- `src/run_sequence_training.py`
- `Data/experiments/day9_sequence_training/fd001_live_check/metrics.json`

Mention:

- RMSE `35.34`
- MAE `25.77`
- R2 `0.56`

## 4:20 - 5:10 Maintenance Scheduling

Explain:

- predictions become useful only when converted into decisions
- the scheduler uses downtime, repair, labor, parts, and risk assumptions
- PuLP solves the maintenance plan
- sensitivity analysis checks robustness under cost changes

Show:

- `src/maintenance_scheduler.py`
- `Data/experiments/week2_checkpoint/week2_smoke/scheduler_summary.md`
- `optimal_schedule.csv`
- `sensitivity_analysis.csv`
- `cost_vs_risk_tradeoff.png`

Mention:

- solver status `Optimal`
- 12 maintenance tasks
- total cost around `$1.07M`

## 5:10 - 6:20 Dashboard

Run:

```bash
streamlit run app/streamlit_app.py
```

Show pages:

- Overview
- Equipment Detail
- Alerts Configuration
- Reports

Explain:

- dashboard reads latest saved artifacts
- fallback data keeps deployment stable
- Docker and hosted Streamlit deployment paths exist

## 6:20 - 7:00 Closing

Summarize:

- data understanding and leakage-safe setup
- anomaly comparison
- RUL forecasting
- maintenance optimization
- dashboard and deployment path

End with:

"The project is not only a model notebook; it is a complete predictive maintenance workflow from sensor data to decisions."
