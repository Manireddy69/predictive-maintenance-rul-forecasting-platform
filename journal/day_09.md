# Day 09

Today was the point where the sequence pipeline had to stop being "data prepared for a future model" and become an actual forecasting workflow.

That meant building a supervised recurrent model that was strong enough to matter, but still constrained enough to understand.

The Day 9 brief was not "use deep learning."

It was:

- train a bidirectional sequence model
- keep the architecture disciplined
- make the attention mechanism inspectable
- tune only the important hyperparameters
- save outputs so the rest of the project can use them

## What I worked on

- added a supervised bidirectional LSTM + attention model
- kept the recurrent stack limited to `2` or `3` bidirectional layers
- enforced dropout in the requested `0.2` to `0.3` range
- built a PyTorch Lightning training path
- added Optuna-based tuning for learning rate, hidden size, window length, dropout, and layer count
- added artifact saving for metrics, predictions, metadata, checkpoints, and Optuna trials
- extended the training flow so it supports both `rul` regression and `failure_in_next_window` classification

## Why this day mattered

Day 8 gave the project sequence windows.

Day 9 had to prove those windows were actually usable in a real supervised forecasting workflow.

This matters because sequence modeling can go wrong in a very believable way:

- the code trains
- the loss decreases
- the architecture sounds impressive
- the predictions are still operationally weak or unreproducible

So the job today was not just "get an LSTM to run."

The real job was to create a training path that is:

- reproducible
- bounded
- testable
- reusable by later project layers

## What got added

The main implementation work lives in:

- `src/sequence_attention_model.py`
- `src/run_sequence_training.py`
- `tests/test_sequence_attention_model.py`

The model path now includes:

- bidirectional LSTM encoding
- temporal attention over sequence outputs
- a dense prediction head
- PyTorch Lightning training loops
- early stopping and checkpointing
- Optuna search over the requested hyperparameters
- evaluation output as a structured prediction frame

## Why the architecture choices were constrained

I intentionally did not make this an open-ended deep learning experiment.

The constraints were useful:

- `2` to `3` recurrent layers is enough to explore capacity without disappearing into architecture tuning
- dropout bounded to `0.2` to `0.3` keeps the implementation aligned with the project requirement
- Optuna only tunes the parameters that materially change model behavior for this stage

That keeps the search space understandable instead of turning it into random guesswork.

## What I checked

I verified the sequence-model test slice with:

- `python -m pytest tests/test_sequence_attention_model.py tests/test_sequence_data.py -q`

The new coverage locked down:

- attention output shape and normalization
- metric computation for both regression and classification
- prediction-frame construction
- Optuna parameter application
- Day 9 architectural constraints

I also ran a small training smoke path through:

- `python -m src.run_sequence_training --fd FD001 --target-mode rul --max-epochs 1 --optuna-trials 0 --disable-prophet-ensemble`

That mattered because the real proof is not only unit tests.

The real proof is:

- the model trains
- the checkpoint saves
- the evaluation pass completes
- the artifacts land in the expected experiment folder

## What I learned

- recurrent modeling becomes much easier to reason about when the sequence-data layer is already disciplined
- attention is useful here because it gives one lightweight interpretability handle without overcomplicating the model
- Optuna helps most when the search space is intentionally narrow
- saving prediction frames early is important because later stages need the outputs, not just the scalar metrics

## What still feels shaky

- how much improvement comes from the recurrent architecture versus the windowing and scaling discipline
- whether a larger hidden size is genuinely useful or mostly a way to overfit faster
- how stable the best hyperparameters will be across other CMAPSS subsets beyond `FD001`
- how much operational value the binary forecasting mode adds compared with pure RUL regression

## Mistakes or traps

- treating a decreasing training loss like proof of a good forecasting model
- tuning too many parameters at once
- saving only summary metrics and not the predictions that later stages need
- letting the architecture grow before the baseline training path is stable

## What exists now

- a supervised bidirectional LSTM + attention training path
- Optuna hyperparameter tuning for the required Day 9 knobs
- saved metrics, checkpoints, predictions, metadata, and trials
- a reusable CLI entrypoint in `src/run_sequence_training.py`
- tests that keep the Day 9 path from drifting

## Day 9 conclusion

Day 9 is complete.

The repo now has a real supervised sequence-forecasting workflow rather than only a prepared sequence dataset.

That changes the project materially because later tasks can now build on:

- saved prediction outputs
- repeatable training commands
- test coverage around the sequence path
- a model that is constrained enough to debug

## Next move

- add a trend and seasonality decomposition layer for the RUL path
- test whether a hybrid forecast improves the final prediction
- keep the forecasting artifacts structured so the scheduler can consume them later
