# LogicVeda Project 1 - Day 1

## Goal
Complete the first day of the project by inspecting the dataset and building the first data-loading pipeline.

## Day 1 tasks
- Confirm the dataset directory and file naming conventions.
- Load one training file from `Data/CMaps`.
- Load the corresponding test file and RUL file.
- Print a summary of the training and test data.
- Inspect the number of units, cycle counts, and sensor statistics.

## What to check
- `train_FD001.txt` contains the training series.
- `test_FD001.txt` contains test cycles without the final failure time.
- `RUL_FD001.txt` contains future remaining useful life values for test units.
- Each row has 26 columns: unit, cycle, 3 settings, and 21 sensor values.

## Next step after Day 1
- Add simple rolling feature engineering in `src/features.py`.
- Create a notebook for visualizing sensor behavior over time.
- Build a baseline anomaly detection model.

## Day 4 planner
- Add an Airflow DAG for batch CSV ingestion, validation, feature engineering, and TimescaleDB loading.
- Validate batch sensor CSVs with Great Expectations checks for schema, nulls, duplicates, and sensor ranges.
- Engineer rolling 1h/8h/24h mean/std/min/max features.
- Engineer lag-1 to lag-12 features.
- Engineer FFT top-5 frequency amplitude features.
- Engineer cross-sensor ratio features and load the result into a TimescaleDB hypertable.
