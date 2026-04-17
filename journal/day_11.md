# Day 11

Today was where the project stopped being only a forecasting pipeline and started acting like a maintenance decision system.

Predictions are useful.

They are not yet a maintenance plan.

So the job today was to take forecast outputs and translate them into a constrained scheduling problem with explicit costs.

## What I worked on

- defined a realistic maintenance cost matrix
- added repair profiles with time, labor, and cost assumptions
- built a PuLP mixed-integer scheduler with binary maintenance-day decisions
- added constraints for technician availability
- added downtime-SLA limits
- added parts inventory limits
- created task-building logic from forecast outputs

## Why this day mattered

This is the day where the project begins to answer the question:

"What would we actually do with the model output?"

That matters because a predictive-maintenance capstone feels incomplete if it stops at:

- a metric table
- a prediction CSV
- a dashboard chart

Those are useful artifacts.

But the business question is closer to:

- which units do we act on
- when do we service them
- what does that schedule cost
- what happens when labor and downtime are limited

## What got added

The core implementation lives in:

- `src/maintenance_scheduler.py`
- `src/week2_checkpoint.py`
- `tests/test_maintenance_scheduler.py`

The scheduling layer now includes:

- a `MaintenanceCostMatrix`
- repair-type profiles such as inspection, minor service, component repair, and major overhaul
- a `SchedulerResources` structure for planning horizon, technician hours, and downtime budget
- PuLP decision variables for assigning a task to a maintenance day
- feasibility constraints that reflect resource limits instead of pretending all tasks can happen immediately

## Why the cost matrix mattered

I did not want the scheduler to be one of those optimization demos that looks sophisticated but is driven by meaningless numbers.

So I used explicit assumptions:

- downtime cost: `$10,000/hour`
- repair cost ranges: `$2,000` to `$15,000` depending on repair type
- technician hourly rate

These are still modeled assumptions.

But they are at least stated assumptions tied to the maintenance problem, not arbitrary coefficients with no interpretation.

## What I checked

I ran the scheduler-focused test slice:

- `python -m pytest tests/test_maintenance_scheduler.py -q`

I also exercised the end-to-end scheduling path through the Week 2 checkpoint runner.

That produced a real optimal schedule and saved artifacts rather than only solving a toy in-memory example.

## What I learned

- the hard part of scheduling is not PuLP syntax, it is defining the task structure and cost logic honestly
- maintenance optimization becomes much more readable once repair types are explicit
- binary scheduling variables are a good fit here because each task needs one concrete maintenance day
- turning forecasts into candidate tasks is as important as the optimizer itself

## What still feels shaky

- how sensitive the chosen repair profile assumptions are to different industrial settings
- whether the current parts inventory abstraction should later become part-specific rather than repair-type-based
- how much better the candidate-task prioritization could become if it included uncertainty bands rather than point RUL predictions

## Mistakes or traps

- building an optimizer before defining a meaningful cost structure
- treating all maintenance actions as interchangeable
- ignoring labor and downtime limits
- feeding too many low-priority tasks into the scheduler and calling the result "optimal"

## What exists now

- a first real maintenance optimization layer
- explicit maintenance cost assumptions
- binary day-assignment scheduling
- resource-constrained feasibility logic
- tests around the scheduler path

## Day 11 conclusion

Day 11 is complete.

The repo no longer stops at forecasting.

It now contains a decision layer that can convert model outputs into a constrained maintenance schedule with interpretable cost structure.

## Next move

- test how stable the schedule is when the cost assumptions move
- save schedule outputs in report-friendly formats
- make the scheduler artifacts easy for the dashboard to consume
