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
