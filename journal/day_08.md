# Day 08

Today was the point where the project had to stop talking about sequence models in theory and start producing real sequence samples.

That sounds like a small preprocessing day.

It is not.

If sequence construction is wrong, then the model can look perfectly healthy while learning from misaligned labels.

That is one of the easiest ways to build a convincing but invalid RUL pipeline.

So the real goal today was:

- build sliding windows cleanly
- keep the split unit-aware
- make the target alignment explicit
- create a reusable batching layer for the next training step

## What I worked on

- added a dedicated sequence-preparation module
- added row-level test RUL expansion from the CMAPSS `RUL_FD001.txt` style file
- added sliding-window generation for both continuous `rul` and binary `failure_in_next_window`
- added a PyTorch Lightning-compatible `DataModule`
- added tests for window shapes, target alignment, split correctness, and batch output
- added a runnable CLI command for inspecting sequence batches without opening a Python shell

## Why this day mattered

Day 8 is where sequence modeling becomes either disciplined or sloppy.

The main trap is subtle:

- a window spans many cycles
- but the target has to come from one clearly defined point

If I label a window using the wrong cycle, then the model is not solving the problem I think it is solving.

That is why I treated target alignment as the main job today, not as a side detail.

## What got added

The main implementation now lives in:

- `src/sequence_data.py`

This file now handles:

- default sequence feature selection from settings and sensors
- row-level train RUL from failure cycle
- row-level test RUL from the provided unit-level test targets
- sliding-window generation
- `rul` regression targets
- `failure_in_next_window` classification targets
- unit-aware train and validation splitting
- train-only scaling
- Lightning-compatible dataloaders for train, validation, test, and predict

I also added:

- `src/run_sequence_preparation.py`
- `tests/test_sequence_data.py`

## The alignment rule I used

Each sequence window is labeled from the **last cycle inside the window**.

That means:

- the input is cycles `t - window_size + 1` through `t`
- the target RUL is the RUL at cycle `t`

For the binary target:

- the label is `1` when the end-of-window RUL is between `1` and the selected prediction horizon
- otherwise it is `0`

In plain language:

the binary target answers:

"after the final timestep in this window, is failure now close enough to count as within the next horizon?"

That is a much cleaner question than vaguely mixing sequence history with future labels.

## Why the DataModule mattered

The sliding windows by themselves are useful, but they are not yet a training pipeline.

The `DataModule` matters because it centralizes:

- dataset setup
- split handling
- scaling
- tensor conversion
- batch loading
- GPU-friendly loader options

That means Day 9 can focus on the actual recurrent model instead of rebuilding data plumbing inside the training script.

## What I checked

I ran the Day 8 sequence-preparation CLI on `FD001` for both targets.

### Continuous RUL mode

Command:

- `python -m src.run_sequence_preparation --fd FD001 --target-mode rul --window-size 30 --stride 5 --batch-size 64`

Observed summary:

- train windows: `2923`
- validation windows: `663`
- test windows: `2079`
- feature count: `24`
- batch shape: `(64, 30, 24)`

The train RUL range printed as:

- `0.0` to `332.0`

### Binary failure-in-next-window mode

Command:

- `python -m src.run_sequence_preparation --fd FD001 --target-mode failure_in_next_window --window-size 30 --stride 5 --prediction-horizon 30 --batch-size 64`

Observed summary:

- train positive windows: `480`
- train negative windows: `2443`
- validation positive windows: `120`
- validation negative windows: `543`
- test positive windows: `65`
- test negative windows: `2014`

That gave me an immediate sanity check that the windowing code was not producing an obviously broken label distribution.

## Test status

The new sequence-preparation test slice passed:

- `python -m pytest tests/test_sequence_data.py -q`
- `5 passed`

I also re-ran the existing anomaly sequence test file:

- `python -m pytest tests/test_anomaly_lstm_autoencoder.py -q`
- `5 passed`

That matters because the project already had sequence logic inside the anomaly track, and I did not want Day 8 improvements to quietly break the older path.

## What I learned

- sequence preparation is really target-definition work disguised as preprocessing
- test-set RUL expansion is easy to describe but important to get exactly right
- the cleanest target rule is to label from the end of the window, not from an ambiguous midpoint or future offset
- unit-aware splitting needs to happen before window generation if I want to avoid leakage
- a `DataModule` is worth adding early because it makes the next training stage much cleaner

## What still feels shaky

- whether `30` is the best starting window length for the first supervised sequence model
- how much performance will change between `30`, `60`, and `120` timesteps
- whether all `24` current features should be kept for the first recurrent baseline
- how imbalanced the binary failure target will become under different horizon choices

## Mistakes or traps

- generating windows first and splitting later
- using unit identifiers across both train and validation windows
- attaching the wrong cycle's RUL to each window
- defining the binary failure target vaguely instead of tying it to a clear prediction horizon
- pushing directly into an LSTM trainer before checking the actual batch outputs

## What exists now

- a reusable sequence preparation module in `src/sequence_data.py`
- a Lightning-compatible `CMAPSSSequenceDataModule`
- a CLI sequence inspection command in `src/run_sequence_preparation.py`
- tests that lock down window alignment and batching behavior
- README examples showing how to run the new Day 8 path

## Day 8 conclusion

Day 8 is complete.

The repo now has a real sequence-data pipeline instead of scattered sequence utilities.

That is important because Day 9 can now start from:

- known window shapes
- known target alignment
- known split behavior
- known batch output

That is a much better position than jumping into an LSTM and hoping the input pipeline is correct.

## Next move

- build the first supervised LSTM or GRU on top of the new `CMAPSSSequenceDataModule`
- start with the continuous `rul` target first
- compare against the best non-sequence baseline instead of evaluating the recurrent model in isolation
