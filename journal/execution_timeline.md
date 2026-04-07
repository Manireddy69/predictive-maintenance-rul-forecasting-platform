# Execution Timeline

This is the working plan for the project after reading the LogicVeda PDF and stripping out the parts that are mostly presentation pressure.

The PDF wants a full production-style system: ingestion, anomaly detection, RUL forecasting, scheduling, dashboard, monitoring, deployment, demo, and report.

That is fine as outer scope.
It is not a good execution order if the goal is to actually understand the problem.

So this plan follows one rule:

solve FD001 properly first, then add layers

If the RUL pipeline is weak, Kafka will not save it.
If the split is leaking, a dashboard will only help you present wrong results more nicely.

## What matters most

These are the parts that cannot be sloppy:

- CMAPSS schema
- train RUL target creation
- unit-aware splitting
- baselines before deep learning
- sequence preparation
- model evaluation and error analysis

If these are weak, the rest is decoration.

## What can wait

These are not bad, but they are bad early priorities:

- Kafka
- TimescaleDB
- Airflow
- MLflow
- Prophet ensemble
- Kubernetes
- cloud deployment

The PDF includes them because it is trying to make the capstone look industry-ready.
That is different from saying they are the first things you should build.

## Phase 1 - Make the data problem real

### Day 1 - Set up the repo and load FD001

What I am really doing:
- get the repo into a clean state
- load `train_FD001`, `test_FD001`, and `RUL_FD001`
- make sure the column schema is correct

What I need to understand:
- one row is not one independent example
- one unit is one engine
- train goes to failure
- test stops before failure

What should exist by the end:
- working repo
- data loader
- first EDA script or notebook
- first journal note

What can go wrong:
- wrong column count
- treating CMAPSS like plain tabular data
- not understanding what the test RUL file means

### Day 2 - Build the RUL target

What I am really doing:
- derive train RUL from each unit's final cycle
- check that it decreases to zero correctly

What I need to understand:
- target construction is part of the model, not a preprocessing detail
- one mistake here poisons everything later

What should exist by the end:
- `add_train_rul()` or equivalent
- plots or tables proving the target makes sense

What can go wrong:
- off-by-one logic
- mixing test target logic into train logic
- using capped RUL before understanding uncapped RUL

### Day 3 - Figure out which sensors are worth caring about

What I am really doing:
- identify constant sensors
- look at actual sensor behavior over time
- stop pretending every sensor matters equally

What I need to understand:
- some sensors are basically useless
- some sensors drift clearly with degradation
- time behavior matters more than a one-line summary table

What should exist by the end:
- EDA notebook
- list of constant columns
- shortlist of stronger candidate sensors

What can go wrong:
- making a lot of plots and learning nothing
- treating correlation heatmaps as the whole EDA story
- keeping constant columns because they look official

### Day 4 - Preprocessing and split logic

What I am really doing:
- create a clean training pipeline without leakage
- split by unit, not by random rows
- scale using train data only

What I need to understand:
- leakage is one of the easiest ways to fake model performance
- grouped splitting is non-negotiable here

What should exist by the end:
- preprocessing code
- stable train/validation split
- selected feature set for baselines

What can go wrong:
- random row split
- fitting scalers on the full dataset
- deciding on fancy features before the split logic is correct

## Phase 2 - Prove simple models first

### Day 5 - Build tabular baselines

What I am really doing:
- start with models that are easy to debug
- set a floor the sequence model must beat

What I need to understand:
- deep learning is not a substitute for disciplined baselines
- if a simple model is competitive, that matters

What should exist by the end:
- dummy baseline
- linear or ridge model
- one tree-based model
- metrics table

What can go wrong:
- skipping the dummy baseline
- reporting only the best model
- changing preprocessing between models and pretending the comparison is fair

### Day 6 - Anomaly detection comparison and sequence justification

What I am really doing:
- compare simple anomaly detectors against a sequence model instead of assuming the deep model should win
- test whether temporal context adds real value over robust statistical scoring

What I need to understand:
- anomaly detection and RUL are separate tasks and should stay conceptually separate
- strong simple baselines are valuable because they tell me whether the deep model is actually earning its complexity

What should exist by the end:
- `z-score` and `MAD` anomaly baselines
- an LSTM autoencoder with reconstruction-error thresholding
- one saved comparison run with ROC-AUC and PR-AUC
- a notebook that reads and visualizes the saved experiment artifacts
- a journal note that explains the result in plain language

What can go wrong:
- moving to LSTM because it sounds advanced
- comparing row-level and window-level results as if they mean the same thing
- hiding the fact that simple methods might outperform the deep model

#### Day 6 completion note

Completed:

- added `z-score` and `MAD` to the anomaly baseline code
- built the LSTM autoencoder comparison pipeline
- saved experiment outputs to `Data/experiments/anomaly_day6/`
- created `notebooks/03_lstm_autoencoder_comparison.ipynb`

Key result from the completed window-level comparison:

- `MAD Distance`: ROC-AUC `0.941561`, PR-AUC `0.939986`
- `Z-Score Distance`: ROC-AUC `0.940698`, PR-AUC `0.939898`
- `LSTM Autoencoder`: ROC-AUC `0.933422`, PR-AUC `0.928510`

What that means:

- the LSTM autoencoder is strong and valid
- but the robust statistical methods are still slightly better on the current synthetic setup
- that makes Day 6 a success, because the comparison is now evidence-based instead of assumption-based

### Day 7 - Improve the baseline without changing the whole project

What I am really doing:
- make targeted improvements, not random experimentation

What I need to understand:
- capped vs uncapped RUL
- sensor subset decisions
- whether simple rolling features help

What should exist by the end:
- one stronger baseline
- experiment notes that explain what changed

Best Day 7 handoff after Day 6:

- do not replace the simpler baselines just because the LSTM exists
- test only a small number of controlled changes
- focus on whether the synthetic anomaly design is rewarding pointwise deviation more than temporal disruption
- improve only what the Day 6 results suggest is actually weak

What can go wrong:
- changing five things at once
- claiming improvement without controlled comparison
- doing feature engineering just to feel busy

## Phase 3 - Move into sequence modeling

### Day 8 - Build sliding windows

What I am really doing:
- turn unit histories into sequence samples
- align each window with the correct target

What I need to understand:
- a sequence sample is not just a reshaped table
- split logic must still be clean
- target alignment errors are common and subtle

What should exist by the end:
- sequence generator
- validation check on window shapes and labels

What can go wrong:
- generating windows first and splitting later
- leaking unit information across splits
- using the wrong cycle's RUL as the label

### Day 9 - Train the first LSTM or GRU

What I am really doing:
- build the first serious sequence model
- keep it small enough to understand

What I need to understand:
- recurrent models add temporal capacity, not magic
- overfitting happens fast here

What should exist by the end:
- first trained recurrent model
- training and validation loss curves
- comparison against the best baseline

What can go wrong:
- using a big architecture too early
- ignoring normalization
- celebrating train loss

### Day 10 - Evaluate the sequence model properly

What I am really doing:
- check whether the sequence model is actually better in useful ways

What I need to understand:
- RMSE and MAE are not enough if the unit-level behavior is poor
- a slightly better average metric can still hide ugly failure-region behavior

What should exist by the end:
- metrics table
- unit-wise prediction plots
- a clear yes/no answer on whether the sequence model beats the baseline

What can go wrong:
- reporting one number and moving on
- not looking at individual units
- assuming lower loss means better operational behavior

### Day 11 - Improve the sequence model carefully

What I am really doing:
- change one thing at a time and see what matters

What I need to understand:
- window length
- hidden size
- dropout
- sensor subset decisions

What should exist by the end:
- best sequence setup so far
- short note on what helped and what did not

What can go wrong:
- turning tuning into gambling
- no record of experiments
- chasing tiny gains without understanding them

## Phase 4 - Add the extra tracks from the PDF

### Day 12 - Start anomaly detection as a separate track

What I am really doing:
- keep anomaly detection separate from RUL so the project does not become conceptually messy

What I need to understand:
- anomaly detection answers a different question than RUL forecasting
- synthetic data is useful for controlled experiments, not as proof

What should exist by the end:
- anomaly baseline notebook
- note on what data source is being used and why

What can go wrong:
- mixing anomaly labels and RUL logic into one unclear pipeline
- using synthetic data as main evidence

### Day 13 - Start maintenance scheduling logic

What I am really doing:
- turn predictions into decisions

What I need to understand:
- prediction is not the final business output
- maintenance planning needs assumptions about cost, downtime, labor, and constraints

What should exist by the end:
- simple cost formulation
- first scheduler notebook or prototype

What can go wrong:
- pretending optimization is meaningful before predictions are stable
- using made-up costs without saying they are assumptions

### Day 14 - Mid-project checkpoint

What I am really doing:
- stop adding and assess what actually works

What should exist by the end:
- cleaned code
- updated timeline
- journal notes
- clear statement of current strengths and weaknesses

## Phase 5 - Build the capstone layers

### Day 15 to Day 17 - Dashboard basics

What I am really doing:
- build a simple interface only after the model side is credible

What should exist by the end:
- basic Streamlit app
- sensor plots
- RUL visualizations

What can go wrong:
- spending more time on charts than on model quality

### Day 18 to Day 20 - Monitoring and retraining ideas

What I am really doing:
- explore drift and monitoring, but only if there is something worth monitoring

What should exist by the end:
- simple drift notebook or prototype
- retraining criteria notes

What can go wrong:
- building a monitoring story around a weak pipeline

### Day 21 to Day 24 - Packaging and deployment basics

What I am really doing:
- make the repo reproducible and shareable

What should exist by the end:
- cleaned repo
- tests for critical code
- deployment basics
- environment cleanup

What can go wrong:
- trying to deploy too much at once
- treating packaging as an afterthought

### Day 25 to Day 28 - Demo and submission package

What I am really doing:
- turn the project into something evaluable

What should exist by the end:
- final README
- screenshots
- result tables
- video
- report PDF
- final QA pass

What can go wrong:
- leaving communication to the end
- having code but no clear story

## Final rule

At every phase, ask:

am I improving the actual predictive maintenance pipeline, or am I just making the repo look bigger?

That question will save time.
