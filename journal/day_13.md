# Day 13

Today was about getting experiment tracking out of the notebook mindset and into something comparative and durable.

The project already had:

- saved artifacts
- metrics files
- multiple model variants

What it still needed was a way to compare those variants cleanly over time.

That is why today focused on MLflow.

## What I worked on

- added MLflow logging for the forecasting pipeline
- logged model, training, and Prophet configuration parameters
- logged metrics such as MAE and R-squared
- added regression calibration metrics
- attached forecast artifacts to MLflow runs
- added variant naming support so runs are easy to compare

## Why this day mattered

Without tracking, tuning becomes memory.

And memory is a bad experiment database.

The project had reached the point where multiple variants existed:

- LSTM-only paths
- hybrid LSTM + Prophet paths
- different training configurations
- different saved runs

That is exactly when MLflow becomes useful instead of decorative.

## What got added

The main work lives in:

- `src/mlflow_tracking.py`
- `src/run_sequence_training.py`
- `tests/test_mlflow_tracking.py`

The tracking path now logs:

- model parameters
- training configuration
- Prophet configuration
- primary metrics
- calibration slope
- calibration intercept
- calibration mean absolute calibration error
- saved artifacts such as metrics JSON, predictions CSV, metadata JSON, summaries, and checkpoints

## Why calibration metrics were worth adding

It would have been easy to stop at:

- RMSE
- MAE
- R-squared

Those are necessary.

They are not the full operational story.

Calibration matters because a forecasting model can have decent average error while still being systematically too optimistic or too pessimistic.

That is especially important in maintenance work, because:

- underestimating urgency is dangerous
- overestimating urgency is expensive

## What I checked

I ran the MLflow smoke path:

- `python -m src.run_sequence_training --fd FD001 --target-mode rul --max-epochs 1 --optuna-trials 0 --disable-prophet-ensemble --run-name mlflow_smoke --variant-name lstm_smoke_v1 --log-mlflow`

That completed successfully and created a real MLflow run plus a local summary artifact under:

- `Data/experiments/sequence_tracking/`

I also ran:

- `python -m pytest tests/test_mlflow_tracking.py -q`

That kept the tracking layer from becoming a "works on my machine" feature.

## What I learned

- experiment tracking becomes valuable the moment variant names start to matter
- calibration metrics are a better addition here than chasing more exotic dashboards too early
- artifact logging is just as important as metric logging because later review often depends on the saved prediction files
- the cleanest MLflow integration is via the existing training CLI, not a separate tracking-only script

## What still feels shaky

- which forecasting variants should become the standard comparison set for final presentation
- how much the calibration story will change once more hybrid runs complete on a stable environment
- whether future tracking should include scheduler-level business metrics in the same MLflow experiment or in a separate one

## Mistakes or traps

- logging runs without meaningful variant names
- saving metrics but not the artifacts needed to inspect model behavior
- pretending experiment tracking matters if the compared configurations are not actually controlled

## What exists now

- local MLflow tracking for forecasting runs
- calibration-aware metric logging
- run summaries saved into project artifacts
- tests around the tracking utilities

## Day 13 conclusion

Day 13 is complete.

The forecasting layer is now trackable, comparable, and easier to explain across multiple variants.

That is a major quality upgrade because the project no longer depends on manual memory to compare model runs.

## Next move

- package the full forecasting-to-scheduling path into one checkpoint flow
- make the checkpoint output strong enough for a mid-project milestone
- prepare a UI layer that can actually read the saved artifacts
