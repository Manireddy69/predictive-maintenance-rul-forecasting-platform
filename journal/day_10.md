# Day 10

Today was about moving from "the LSTM predicts RUL" to "the forecasting stack can separate temporal pattern types and blend them intentionally."

That is where the Prophet integration came in.

The requirement was not to replace the sequence model.

It was to add:

- trend and seasonality decomposition
- a hybrid ensemble
- a fixed final weighting of `0.7 * LSTM + 0.3 * Prophet`

## What I worked on

- added a Prophet baseline on cycle-level RUL aggregates
- extracted trend and seasonality components from the Prophet forecast
- blended the final RUL output as `0.7` LSTM and `0.3` Prophet
- added hybrid metrics and hybrid prediction columns to the saved outputs
- wired the Prophet options into the sequence training CLI
- added a timeout-protected Prophet worker so the Windows environment would not stall the entire run

## Why this day mattered

The LSTM is good at nonlinear sequence modeling.

Prophet is useful here for a different reason:

- it gives an explicit trend component
- it gives an explicit seasonality component
- it provides a second forecasting view that is easier to describe operationally

That means the hybrid path is not just "another model."

It is a way to make the forecasting stack less monocultural.

## The difficult part

The tricky part was not the weighted average.

The tricky part was the local runtime behavior.

On this Windows machine, the Prophet fit step would sometimes stall inside the Stan backend.

That creates a bad failure mode:

- the forecasting run starts
- the LSTM part finishes
- the whole workflow hangs during hybrid evaluation

That is unacceptable for the later scheduler pipeline.

So the real Day 10 fix was not only the hybrid model.

It was making the hybrid path safe.

## What got added

The main work lives in:

- `src/sequence_attention_model.py`
- `src/prophet_worker.py`
- `src/run_sequence_training.py`
- `src/run_week2_checkpoint.py`

The forecasting path now supports:

- Prophet training-frame construction from cycle-aligned RUL targets
- Prophet trend and seasonality extraction
- weighted hybrid predictions
- timeout-controlled Prophet execution in a subprocess
- fallback to LSTM-only predictions when Prophet cannot finish

## Why the fallback mattered

This was one of those places where purity would have been a mistake.

In theory, the strictest version is:

- if Prophet fails, the whole run fails

In practice, that would make the forecasting layer fragile and would block the scheduler, dashboard, and checkpoint work.

So I added a controlled fallback:

- if Prophet finishes, use the hybrid output
- if Prophet stalls, mark the run as fallback and keep the LSTM path alive

That is much better than pretending the hybrid is stable when the environment says otherwise.

## What I checked

I ran a smoke command that intentionally exercised the timeout path:

- `python -m src.run_sequence_training --fd FD001 --target-mode rul --max-epochs 1 --optuna-trials 0 --run-name hybrid_timeout_smoke --prophet-fit-timeout-seconds 5`

The important result was not the metric value.

The important result was:

- the run completed
- it did not hang
- the output recorded `ensemble_applied: 0.0`
- the output recorded `prophet_fallback_used: 1.0`

That proved the hybrid code no longer blocks the whole project when the local Prophet runtime misbehaves.

## What I learned

- decomposition features are useful, but reliability matters more than elegance
- hybrid models need operational failure handling, not just mathematical blending
- a subprocess boundary is sometimes the cleanest way to contain third-party runtime issues
- environment-safe forecasting is more valuable than a theoretically nicer but brittle implementation

## What still feels shaky

- whether the Prophet component will behave differently across other deployment targets
- how much true improvement the hybrid will show on datasets beyond the local smoke path
- whether the fixed `0.7 / 0.3` weighting is still best after broader comparison

## Mistakes or traps

- assuming a library is production-safe just because the import works
- letting a secondary model block the full forecasting pipeline
- blending outputs without saving the component predictions separately
- hiding fallback behavior instead of recording it explicitly

## What exists now

- cycle-level Prophet decomposition for the RUL path
- hybrid LSTM + Prophet prediction support
- timeout and strict-mode controls in the CLIs
- a subprocess worker that keeps the hybrid path from freezing the run
- prediction outputs that include LSTM, Prophet, hybrid, trend, and seasonality columns when available

## Day 10 conclusion

Day 10 is complete.

The forecasting stack now does more than train one recurrent model.

It can:

- build a structured decomposition baseline
- produce a hybrid forecast
- survive local Prophet issues without freezing the project

That is a much stronger position for the scheduler and dashboard work.

## Next move

- convert forecast outputs into maintenance decisions
- define realistic cost assumptions instead of abstract model-only metrics
- build the first optimization-based scheduler around the forecast artifacts
