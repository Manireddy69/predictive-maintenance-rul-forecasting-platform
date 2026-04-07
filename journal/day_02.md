# Day 02

Today was the day the project stopped being "some notebooks plus a dataset" and started feeling like a system.

Once I knew I was dealing with time-based sensor data, the next question became obvious:
where should this data actually live if I want ingestion, querying, feature engineering, and later model work to make sense together?

That is why Day 2 was about TimescaleDB and schema design.

## What I worked on

- set up TimescaleDB locally with the option to move to AWS RDS later
- designed the main telemetry schema
- created a hypertable for time-series sensor data
- thought about chunk intervals and compression policy
- decided what the primary indexed columns should be

## Why TimescaleDB made sense here

I did not want to treat telemetry like generic application data.

The project has:

- timestamps
- equipment IDs
- high-frequency sensor readings
- future need for windowed aggregations
- later need for time-based retention and compression

A normal relational table can store that.
That is not the same as saying it is the best fit.

I used `TimescaleDB` because it is still PostgreSQL underneath, but it gives better time-series behavior through:

- hypertables
- chunking by time
- compression policies
- time-based query patterns that fit telemetry much more naturally

That made it a better foundation than pretending everything was just another CRUD table.

## How I approached the storage decision

I tried to keep the setup practical instead of overdesigned.

The question was not:
"what is the most impressive cloud architecture I can mention?"

The question was:
"what database choice makes later ingestion and feature work easier without creating complexity I do not yet need?"

That is why the answer was:

- start locally so I can control the system and understand it
- keep AWS RDS as a realistic future path
- use a schema that is simple enough to query and extend

## What I put into the schema and why

The core columns were:

- `timestamp`
- `equipment_id`
- `run_id`
- 20+ sensor fields
- `failure_label`

The sensor set included things like:

- `vibration_x`
- `vibration_y`
- `vibration_z`
- `temperature`
- `pressure`
- `current`
- `acoustics`

Why this structure:

- `timestamp`
  This is the actual spine of the data.
  Without it, there is no real time-series story.

- `equipment_id`
  Needed to separate one machine from another.
  This becomes critical for both aggregation and model grouping later.

- `run_id`
  Useful when one equipment stream has multiple sessions, simulated runs, or batch loads.
  It prevents the data from becoming one long anonymous timeline.

- sensor columns
  Wide schema is acceptable here because the project is sensor-centric and the set is known enough to model directly.

- `failure_label`
  Even if it is sparse or delayed, it matters because later anomaly and maintenance logic need some operational target or event reference.

## Why the index design mattered

I treated the `timestamp` as the primary time axis because all later operations depend on time windows.

The combined indexing logic around time and equipment matters because later questions will look like:

- what happened on one machine in the last hour?
- what were the most recent vibration values before a failure?
- what features should I engineer over rolling windows?

If the schema makes those queries awkward, then the ML pipeline gets clumsy too.

## Why I thought about chunk intervals and compression now

This was not about premature optimization.

It was about making sure the storage model matches the data rhythm.

I used the idea of:

- recent data in smaller chunks
- older data compressed into larger chunks

The 7-day to 1-month thinking came from the fact that recent telemetry is usually queried more interactively, while older telemetry is more often historical context.

That is useful because:

- recent data stays more accessible
- historical data stops becoming a storage burden
- the system still feels like one database instead of one live table plus one archive hack

## What changed in my head

Before Day 2, I was still half-thinking like a notebook person.

Meaning:
- load files
- transform them in memory
- save outputs somewhere later

After Day 2, the thinking shifted to:

- telemetry needs a home
- that home needs to respect time
- schema design is not admin work
- schema design shapes what feature engineering and monitoring will feel like later

That was a useful shift.

## Why this matters for the algorithm side

Even though this day was "database work," it still connects directly to modeling.

- rolling features from Day 4 depend on timestamped storage
- anomaly context depends on grouping by equipment and time
- failure labeling becomes easier to reason about when run boundaries are explicit
- any future online scoring path depends on the storage design being sane

So this was not separate from ML.
It was infrastructure in service of ML.

## What still felt shaky

- the best chunk interval for the data volume I will actually end up with
- whether the schema should stay wide or later move to a more normalized sensor-event structure
- how much of the final project will use CMAPSS-style batch data versus streaming-style telemetry

## Mistakes I wanted to avoid

- designing a schema that looks enterprise-ready but is miserable to query
- adding cloud complexity before the local shape is clear
- ignoring run/session boundaries and regretting it later
- treating database design like a side issue unrelated to modeling

## What I am taking from Day 2

Day 2 gave the project a place to stand.

Without that, the later streaming and batch work would just be scripts tossing CSVs around.
With it, the project starts to behave like an actual telemetry pipeline.

## Next move

Once the storage model felt real, the next natural step was:

- create a stream
- push sensor events through it
- stop thinking only in terms of static files
