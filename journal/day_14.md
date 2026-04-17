# Day 14

Today was the checkpoint day.

That meant stopping the feature-by-feature build mindset and making sure the pieces actually connect.

The requirement here was not another isolated module.

It was an end-to-end slice:

- data
- features
- prediction
- maintenance schedule

## What I worked on

- built a Week 2 checkpoint runner that can use existing predictions or train a fresh model
- connected the forecasting artifacts to maintenance candidate generation
- connected candidate generation to the scheduler and sensitivity analysis
- added a checkpoint summary artifact that bundles the core outputs
- verified the batch simulation path from forecast outputs into scheduling artifacts

## Why this day mattered

A project can feel more complete than it really is when each module works separately.

The checkpoint day is where that illusion gets tested.

What mattered was not:

- whether the model still trains
- whether the scheduler still solves

What mattered was:

- can the outputs of one stage actually drive the next stage without manual cleanup

That is the difference between a collection of scripts and a working pipeline slice.

## What got added

The checkpoint layer lives in:

- `src/week2_checkpoint.py`
- `src/run_week2_checkpoint.py`

The checkpoint flow now supports:

- reading the latest saved forecast artifacts automatically
- optionally training a fresh forecast model
- selecting the highest-risk candidate assets
- solving the maintenance schedule
- running sensitivity analysis
- writing a summary JSON that ties the main results together

## What I checked

I ran the checkpoint smoke path and verified the saved outputs under:

- `Data/experiments/week2_checkpoint/week2_smoke/`

The meaningful part of that run was not a single number.

It was that the project successfully produced:

- candidate maintenance tasks
- an optimal schedule
- sensitivity scenarios
- a trade-off figure
- a summary JSON that the dashboard can read

That is the first real end-to-end Week 2 checkpoint.

## What I learned

- the handoff between forecasting and scheduling needs its own code, not just shared assumptions
- selecting candidate assets is a necessary bridge between "all predictions" and "actionable schedule"
- batch simulation is easier to explain once the outputs are bundled into a single checkpoint folder
- checkpoint runners are useful because they expose whether the interfaces between modules are actually clean

## What still feels shaky

- how aggressively candidate filtering should happen before optimization
- whether the checkpoint runner should later support richer forecasting uncertainty inputs
- how much the checkpoint summary should expand once dashboard reporting becomes more mature

## Mistakes or traps

- assuming the scheduler can consume raw prediction outputs without a translation layer
- making end-to-end runs depend on manual file selection every time
- declaring a checkpoint complete without saved summary artifacts

## What exists now

- a Week 2 checkpoint runner
- a forecast-to-schedule batch simulation path
- structured checkpoint outputs for later UI and reporting use

## Day 14 conclusion

Day 14 is complete.

The project now has a working checkpoint prototype instead of disconnected forecasting and optimization pieces.

That matters because later presentation and dashboard layers can now sit on top of a real end-to-end workflow.

## Next move

- add the first dashboard shell
- read the saved checkpoint outputs into a UI
- make the project easier to inspect without running every script manually
