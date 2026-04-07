# Day 03

Today the project finally started moving.

Up to this point, most of the work was about data understanding and system shape.
That mattered, but it still felt a little static.

Day 3 changed that by introducing streaming.

The moment I started simulating live sensor flow, the project stopped feeling like "historical analysis" and started feeling like predictive maintenance.

## What I worked on

- set up Kafka with a single broker and Zookeeper
- created the topics:
  - `raw-sensor-data`
  - `cleaned-features`
  - `anomalies-flagged`
- wrote a Python producer that simulates live equipment telemetry
- injected anomalies at random intervals into the stream

## Why I used Kafka here

I did not add Kafka because the project PDF said "streaming" and I wanted to check the box.

I used it because once the project includes:

- sensor readings over time
- downstream feature computation
- anomaly detection
- possible future consumers like dashboards or alerting

then a streaming backbone becomes a reasonable abstraction.

Kafka made sense because it lets the project separate concerns:

- one process can produce raw sensor events
- another can clean or enrich them
- another can score anomalies
- another can store results

That is cleaner than building one giant Python script that tries to do everything in one loop.

## Why the topics were split this way

I wanted each topic to represent a stage in the pipeline rather than a random naming convention.

### `raw-sensor-data`

This is the untouched or near-raw telemetry stream.

Why it matters:

- it is the closest thing to the source
- it gives me a clean place to debug the producer
- it keeps raw ingestion separate from feature logic

### `cleaned-features`

This is where the data becomes more useful for downstream models.

Why it matters:

- not every downstream step should need to repeat cleaning
- it separates "data quality work" from "model scoring work"
- later consumers can subscribe to something more stable than raw noisy events

### `anomalies-flagged`

This topic exists because anomaly output is not the same thing as raw telemetry.

Why it matters:

- alert-like messages usually need their own stream
- consumers of anomaly events should not have to reconstruct them from raw values
- it creates a clearer path to dashboards or notifications later

## How I approached the producer

I did not want the producer to just emit random numbers every second.

If I did that, the stream would look active, but it would not mean much.

So the producer had to simulate:

- mostly normal operation
- continuity over time
- anomaly injection at controlled random intervals

That balance matters.

If everything is too clean, anomaly detection becomes artificial.
If everything is too chaotic, then the stream stops resembling equipment behavior and the downstream models learn noise.

## Why I used injected anomalies in the stream

The project needs anomaly detection, but most real maintenance settings do not hand you perfect event labels at the exact moment you want them.

So the synthetic anomaly injection was not a shortcut.
It was a controlled evaluation strategy.

It let me ask:

- can the downstream system tell when the stream behavior changes?
- do certain anomaly types stand out more than others?
- what does "abnormal" look like in motion, not just in a static CSV?

## What changed in my head

Before Day 3, I was still mostly thinking in batch.

Meaning:

- load a file
- transform it
- save it

After Day 3, the project started to feel event-driven.

That changes how I think about everything:

- validation has to happen during flow, not only after the fact
- feature engineering can become streaming-aware or batch-aware
- anomaly detection is now connected to timing, not just labels
- storage and computation are now part of a pipeline, not isolated steps

## Why the streaming setup matters for later algorithm work

This day still connects directly to modeling.

- Day 4 feature engineering needs a stream or staged batches to act on
- anomaly outputs later need somewhere to go
- if the raw topic is messy, downstream model behavior becomes harder to trust
- if the topic separation is clear, comparing raw vs cleaned vs flagged behavior becomes much easier

So even though Kafka looks like "systems work," it is really part of making the anomaly and maintenance story real.

## What still felt shaky

- whether one broker is enough even for realistic local testing
- how much event schema enforcement I should add now versus later
- whether the anomaly injection timing is too obvious for downstream detectors

## Mistakes I wanted to avoid

- building a producer that creates noise instead of telemetry
- putting too much logic into the producer and making it impossible to reason about
- using one topic for everything and losing pipeline clarity
- adding Kafka just for optics without using it meaningfully

## What I am taking from Day 3

Day 3 gave the project movement.

Now the repo is not just:
"here is some historical data and a model."

It is becoming:
"here is a flow of telemetry, and here are the stages where data gets cleaned, transformed, scored, and stored."

That is a much better foundation.

## Next move

Once the stream existed, the next step was obvious:

- build the batch/validation path properly
- engineer features in a disciplined way
- connect the raw data flow to something model-ready
