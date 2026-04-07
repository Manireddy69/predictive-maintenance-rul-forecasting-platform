# Day 01

Today was less about "starting the project" and more about stopping myself from choosing data too casually.

It would have been easy to grab one dataset, call it the project dataset, and move on.
That would have made the repo look cleaner, but it would have made the thinking weaker.

The real problem on Day 1 was:
what kind of data do I need for predictive maintenance, and what role should each source play?

## What I worked on

- selected the NASA CMAPSS turbofan dataset as the core sequential dataset
- looked at Kaggle predictive maintenance datasets as a simpler tabular and sanity-check reference
- set up synthetic data generation so I could inject controlled anomalies instead of waiting for "perfect labels"
- started the first EDA notebook
- checked distributions, correlations, missing-value patterns, stationarity behavior, and FFT signatures

## Why I used more than one dataset

I did not want to overload one dataset with every responsibility.

I used each source for a different reason:

- `NASA CMAPSS`
  This is the serious sequential dataset.
  It is the right place to understand degradation, engine life, and time-aware modeling.

- `Kaggle predictive maintenance datasets`
  These are useful because they are easier to inspect quickly.
  They help me sanity-check schema ideas, sensor naming patterns, and maintenance-style labeling conventions.

- `Synthetic data`
  I needed controlled abnormal behavior for anomaly detection experiments.
  Real maintenance data rarely gives clean anomaly labels in the exact form I want.
  Synthetic generation lets me decide what "abnormal" means and test detectors more honestly.

## How I approached the synthetic data

I did not want fake data that looks obviously fake.

So the goal was not:
"generate random numbers and call them telemetry."

The goal was:
"create something with enough structure that anomaly detection methods have to work for a reason."

I used:

- `Faker` for realistic-looking identifiers and event context
- controlled random walk behavior so signals drift over time instead of jumping aimlessly
- injected anomaly patterns at chosen intervals so I could later evaluate whether a detector caught something real or just noise

Why random walk matters:

sensor systems in real equipment usually have continuity.
They do not teleport between unrelated values every second.
If I generate totally independent rows, a lot of anomaly methods will look better or worse for the wrong reason.

## What I looked at in EDA and why

### Distribution analysis

I needed to see whether the sensor variables were narrow, heavy-tailed, skewed, or obviously unstable.

That matters because method choice depends on this.
If a feature is badly skewed or has strong tails, mean-and-standard-deviation style methods can become fragile.

### Correlation heatmap

I used this to see whether sensors were moving together or just sitting beside each other in the table with no useful relationship.

It is not the whole EDA story, but it helps identify:

- redundant sensors
- groups of sensors that may reflect the same physical behavior
- where cross-sensor ratios might later make sense

### Stationarity tests: `ADF` and `KPSS`

I did not want to use these just because they look academic.

I used them because predictive maintenance data often changes over time by design.
Degradation means "the process is evolving," so I needed to understand whether the time series looked approximately stable or obviously non-stationary.

Why both tests:

- `ADF` is useful because it tests for a unit-root style non-stationarity
- `KPSS` is useful because it comes from the opposite angle and tests whether the series is stationary around a level or trend

Using both gave me a more honest read than relying on one test and pretending the answer was settled.

### Missing-value patterns

Missingness in telemetry is not just a cleaning issue.
It can mean sensor dropouts, logging gaps, or equipment-specific behavior.

I wanted to see:

- are values missing randomly?
- are some sensors weak or unreliable?
- are there blocks of missingness that will matter later for streaming or feature engineering?

### FFT spectrum analysis

I included FFT early because vibration-style data is not only about level and trend.
Frequency behavior matters too.

If a signal has strong periodic components, then the time-domain summary alone can hide useful structure.
That is why frequency features later made sense in the batch feature pipeline.

## What changed in my head

At the start of the day, it was tempting to think:
"Day 1 is just data loading plus a few charts."

That is not really what happened.

By the end of the day, the project looked more like this:

- CMAPSS is the main sequence-learning dataset
- Kaggle data is supporting context, not the whole story
- synthetic data is not fake decoration, it is an evaluation tool
- EDA is not there to produce pretty plots
- EDA is there to justify later modeling choices

That shift matters.

## Why these choices matter later

The methods from later days did not come from nowhere.

- `FFT` on Day 1 supports the engineered frequency features from Day 4
- stationarity checks help explain why naive assumptions can fail on sensor sequences
- synthetic anomaly generation is what makes Day 5 and Day 6 anomaly evaluation possible
- multiple data sources prevent the whole project from becoming too dependent on one imperfect dataset

## What still felt shaky

- how much trust to place in Kaggle datasets for anything beyond support and comparison
- which anomaly patterns should be injected first: spikes, drift, dropout, or oscillation changes
- how much of the early EDA should stay in notebooks versus move into reusable code

## Mistakes I wanted to avoid

- pretending one dataset can answer every project question
- doing EDA that has no modeling consequence
- generating synthetic anomalies that are too easy and then claiming strong detection performance
- treating stationarity tests as absolute truth instead of one piece of evidence

## What I am taking from Day 1

Day 1 was really about framing the data problem properly.

If I choose the wrong data roles, then later model comparisons become muddy.
If I understand the role of each dataset clearly, the rest of the week becomes much more coherent.

## Next move

Next I needed to stop thinking only about files and start thinking about system shape:

- where the sensor data should live
- how time-based storage should work
- what schema will support both analytics and modeling later
