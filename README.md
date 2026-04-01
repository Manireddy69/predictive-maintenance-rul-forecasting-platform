# LogicVeda Project 1

Welcome to the LogicVeda Project 1 workspace. This project is built around the Predictive Maintenance and RUL Forecasting use case from the LogicVeda capstone PDF.

## What this folder is for
This repository is your working area for learning by doing. The goal is to turn the project plan into real code, step by step.

## Why this project is valuable
You will learn how to build a complete data science system that:
- ingests and prepares equipment sensor data
- detects anomalies in real time
- predicts remaining useful life (RUL)
- recommends maintenance schedules
- presents results through a dashboard
- is ready for deployment with Docker

## How the workspace is organized
- `src/` - the core Python code for data loading, preprocessing, feature engineering, models, and training
- `app/` - the interface/dashboard or demo application
- `deploy/` - deployment files such as Docker setup and environment configs
- `tests/` - small tests to verify that each piece works correctly
- `notebooks/` - exploratory work, experiments, and analysis
- `Data/` - the dataset files already provided in this project

## Getting started
1. Open `Data/` and inspect the dataset files.
2. Run `day1_eda.py` from the project root to inspect the dataset structure and Day 1 RUL assumptions.
   - `python day1_eda.py --source nasa`
   - `python day1_eda.py --source synthetic`
   - `python day1_eda.py --source kaggle --kaggle-name your_dataset_name`
3. Start with a small data-loading script in `src/data.py`.
4. Use Day 1 to answer the important questions first: what the target is, which sensors are constant, and how leakage can happen.
5. Add feature engineering only after the sequence structure and target logic are clear.
6. Move into anomaly detection and forecasting after the data understanding is stable.

## A friendly workflow
1. Read the project PDF and keep the main goals in mind.
2. Work in small increments: one feature, one model, one evaluation at a time.
3. Use notebooks for exploration and `src/` for reusable code.
4. Keep notes on what you learn and what changes next.

## Suggested first steps
1. Start with `Data/CMaps` because it is the main run-to-failure dataset for RUL learning.
2. Use `src/data.py` to understand the CMAPSS schema: unit, cycle, settings, and 21 sensors.
3. Run `day1_eda.py` and the Day 1 notebook to identify constant columns, high-variance sensors, and rough degradation candidates.
4. Make sure you understand how train RUL is derived before engineering any features.
5. Only then move to rolling features and a baseline model.

## Keep it simple and practical
This project is about learning the full flow, not about making it perfect on the first try. Start with easy, working code, then improve it from there.

If you want, I can also create a starter scaffold with `src/data.py`, `src/features.py`, and a sample notebook to get you moving quickly.
