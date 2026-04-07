# Day 04

Today was the first day the project felt operational instead of exploratory.

The question shifted from:
"what kind of data and pipeline do I want?"

to:

"can I move a batch of telemetry through validation, feature engineering, and storage in one controlled flow?"

That is why Day 4 centered on the Airflow DAG and feature engineering.

## What I worked on

- built the first Airflow DAG for batch ingestion
- wired the flow:
  - batch CSV ingestion
  - Great Expectations validation
  - feature engineering
  - TimescaleDB load
- engineered rolling statistics
- added lag features
- added FFT-based frequency features
- added cross-sensor ratio features

## Why I used Airflow here

I did not want to keep running pipeline steps manually and pretending that counted as orchestration.

The project already had enough moving parts:

- a batch data source
- validation checks
- feature creation
- database loading

At that point, a DAG made more sense than a loose collection of scripts.

I used `Airflow` because it gave me:

- ordered task execution
- visible dependencies
- a natural place to add retries and logging later
- a more realistic pipeline shape for the capstone

The important part is that I kept the DAG narrow.
It is one pipeline slice, not a giant orchestration monster.

## Why I used Great Expectations

Validation needed to become explicit.

Without that, the flow becomes:

- load CSV
- hope it is fine
- engineer features
- discover problems too late

That is a weak pattern.

I used `Great Expectations` because it makes validation rules visible.
Instead of silently assuming schema correctness, I can state things like:

- required columns exist
- sensor fields are numeric
- nulls are limited or absent where they should be
- duplicates are not slipping through
- ranges are plausible

That matters because bad telemetry ruins downstream features quietly.

## How I approached the feature engineering

I tried to be deliberate here.

The question was not:
"how many features can I generate?"

The question was:
"what kinds of behavior should the features capture?"

So I split the feature logic by what signal it is trying to represent.

### Rolling 1h / 8h / 24h mean, std, min, max

Why I used these:

- rolling `mean` gives local level
- rolling `std` gives volatility
- rolling `min` and `max` give short-range extremes

Why multiple windows:

- `1h` catches short-term behavior
- `8h` gives medium context
- `24h` gives slower operational movement

This matters because one window size cannot explain all temporal behavior.

### Lag-1 to lag-12

Why I used lags:

lag features let the model or analysis compare the current reading to recent history directly.

They are simple, but useful for:

- short-term memory
- direction of change
- detecting local discontinuity

I used a series of lags instead of one because a single lag can miss the shape of recent movement.

### FFT top 5 frequency amplitudes

Why I used FFT here:

some sensor behaviors, especially vibration-like signals, are not fully described by level or trend.
Frequency structure matters.

Taking the top amplitudes gave me a compact summary of periodic content without storing the full spectrum every time.

This was one of those cases where Day 1 EDA helped justify Day 4 engineering.

### Cross-sensor ratios

Why I used ratios:

not all abnormal behavior appears in one sensor alone.
Sometimes the relationship between two sensors changes first.

Ratios can capture that kind of relative behavior more clearly than the raw values themselves.

I liked them here because they are interpretable.
If a ratio changes sharply, that usually tells a clearer operational story than an abstract latent feature.

## What changed in my head

Before Day 4, feature engineering still felt like something that happens in notebooks.

After Day 4, it became much more concrete:

- ingestion needs validation before feature creation
- features are part of the pipeline, not side experiments
- database loading should happen after the features are stable, not before
- orchestration is useful when it reflects real dependencies

That made the repo feel more disciplined.

## Why the algorithm choices matter

The feature set was not random.
Each type was chosen to capture a different failure-relevant pattern:

- rolling stats for state and volatility
- lags for short-term memory
- FFT amplitudes for periodic behavior
- ratios for cross-sensor relationships

That is a more honest feature-engineering story than just:
"I made a lot of columns and hoped one would help."

## What still felt shaky

- how much feature redundancy I just introduced
- whether all three rolling windows are equally useful
- whether the ratio set should stay broad or become curated later
- how much the engineered features will help anomaly detection versus RUL separately

## Mistakes I wanted to avoid

- building a DAG that only looks impressive but is hard to debug
- validating too late
- generating many features with no explanation for why they exist
- confusing "more engineered columns" with "better information"

## What I am taking from Day 4

Day 4 made the pipeline real.

The repo now had a path from raw batch input to validated engineered data inside a time-series store.
That is a meaningful transition.

It also gave the later anomaly work a stronger foundation because the data flowing into the detectors now has more context than raw sensor values alone.

## Next move

Once the feature pipeline existed, the next step was to ask:

- what is the simplest anomaly baseline that deserves to be beaten?
- how do I evaluate it fairly on controlled anomalies?
