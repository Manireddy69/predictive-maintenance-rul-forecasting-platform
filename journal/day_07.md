# Day 07

Today was a checkpoint day.

That sounds less exciting than model-building days, but it matters for a reason.

By Day 7, the project had enough moving pieces that it was easy to say:
"a lot is happening."

That is not the same as saying:
"the first week produced a clear, defensible result."

So the goal today was to make Week 1 legible.

## What I worked on

- generated a committed Week 1 EDA report
- closed the Kafka to TimescaleDB gap by adding the consumer path
- added MLflow logging for anomaly comparison runs
- wrote a Week 1 checkpoint summary
- checked the anomaly and streaming-related test slice

## Why this day mattered

A checkpoint is where the project has to stop sounding good and start reading clearly.

That means I needed proof for three things:

- EDA was not just trapped inside notebooks
- Kafka to TimescaleDB was not just a producer with no ingestion path
- anomaly model comparison was not just terminal output with no tracking story

That is what Day 7 fixed.

## EDA report

I created a generated markdown EDA report and committed it into the repo.

Why this mattered:

- it turns the Day 1 analysis into an actual artifact
- it gives the repo one place to see dataset shape, variability, constant sensors, stationarity, and FFT snapshots
- it makes the Week 1 checkpoint easier to review without opening notebooks first

The report is at:

- `reports/week1_eda_report.md`

## Kafka to TimescaleDB

This was the most important systems gap I found today.

Before today:

- there was a Kafka producer
- there was a TimescaleDB ingestion path for mapped data
- but there was not a direct consumer path from Kafka into TimescaleDB

That meant saying "Kafka to TimescaleDB pipeline" would have been a little too generous.

So I added the consumer side.

Now the path is:

- stream producer sends telemetry to `raw-sensor-data`
- consumer reads the messages
- messages are normalized into the telemetry schema
- rows are inserted into `telemetry.sensor_readings`

That made the checkpoint claim honest.

## Why MLflow was the right addition

By the end of Day 6, anomaly comparison already existed.
But comparison results living only in saved CSV files or terminal output is not the cleanest tracking story.

I added MLflow because it gives the experiment layer:

- a run identity
- parameter logging
- metric logging
- artifact logging

That matters because anomaly detection is already at the point where multiple runs and comparisons should be traceable, not just rerun and remembered.

## How I approached MLflow

I did not want MLflow to become a heavy dependency that blocks normal local work.

So I treated it as an optional logging layer:

- the anomaly comparison still works without it
- when enabled, the run logs metrics and artifacts cleanly
- the local default uses a file-based tracking directory

That felt like the right balance between functionality and simplicity.

## What got logged

The Week 1 anomaly logging now captures:

- run configuration
- best model
- ROC-AUC and PR-AUC per method
- reconstruction threshold when applicable
- saved CSV and markdown artifacts

That makes the checkpoint more than just:
"I compared some models once."

## What the checkpoint says about the algorithms

Week 1 did not end with a flashy deep-learning win.

It ended with a better result than that:

- simple robust methods like `MAD` and `z-score` are genuinely strong
- `LOF` is also a credible baseline
- the `LSTM Autoencoder` is valid and competitive
- but the deep model is not automatically better

That is a good Week 1 result because it is honest and useful.

## Validation status

The current test slice for the anomaly and stream-consumer work passed:

- `12 passed`

That matters because this day was about turning scattered progress into something more stable.

## What changed in my head

Before Day 7, Week 1 felt like a good amount of work.

After Day 7, Week 1 feels like a coherent checkpoint.

That is a big difference.

One is activity.
The other is a project state.

## What exists now

- committed EDA report
- Kafka producer
- Kafka consumer to TimescaleDB
- batch DAG to TimescaleDB
- anomaly baselines
- LSTM autoencoder comparison
- saved experiment artifacts
- MLflow logging path
- Week 1 checkpoint report

## Week 1 conclusion

Week 1 is complete.

The repo now has a credible first predictive-maintenance slice with:

- data understanding
- telemetry storage
- streaming and batch flow
- feature engineering
- anomaly detection comparison
- experiment tracking

That is a much stronger place to stand than "the project has started."
