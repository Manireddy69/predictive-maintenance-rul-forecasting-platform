# Day 12

Today was about stress-testing the scheduler instead of admiring the first optimal result.

That mattered because one optimal schedule under one fixed set of costs is not enough.

The important question is:

does the plan stay reasonable when the economic assumptions change?

## What I worked on

- added scheduler sensitivity analysis
- varied downtime, labor, and repair-cost assumptions across scenarios
- measured schedule robustness against the base plan
- saved the optimal schedule as JSON and CSV
- saved sensitivity outputs as tabular artifacts
- generated a cost-versus-risk trade-off visualization

## Why this day mattered

Optimization outputs can look authoritative even when they are fragile.

If a tiny change in cost assumptions completely reshuffles the plan, then the schedule is not yet operationally trustworthy.

So the real Day 12 job was to ask:

- what happens if downtime becomes more expensive
- what happens if labor becomes more expensive
- what happens if repair parts become more expensive

That makes the scheduler much more useful than a single "best" answer.

## What got added

The sensitivity and reporting work lives mainly in:

- `src/maintenance_scheduler.py`
- `src/week2_checkpoint.py`

The outputs now include:

- `optimal_schedule.json`
- `optimal_schedule.csv`
- `sensitivity_analysis.csv`
- `cost_vs_risk_tradeoff.png`
- `maintenance_candidates.csv`
- `week2_checkpoint_summary.json`

## What I checked

I ran the Week 2 checkpoint smoke path using saved forecast outputs:

- `python -m src.run_week2_checkpoint --predictions-csv Data/experiments/day9_sequence_training/fd001_live_check/test_predictions.csv --run-name week2_smoke --planning-horizon-days 7 --technician-hours-per-day 32 --max-daily-downtime-hours 16 --max-candidate-assets 12`

The run completed with:

- solver status: `Optimal`
- task count: `12`
- total cost: about `1,066,287.97`
- total direct cost: `984,000.0`
- total risk cost: about `82,287.97`

Those numbers matter less than the bigger outcome:

- the scheduler solved
- the artifacts were saved
- the sensitivity outputs were usable by later layers

## What I learned

- schedule robustness is much easier to discuss when the outputs are saved as explicit scenarios
- JSON plus CSV is the right combination here because JSON is good for structured summaries and CSV is better for inspection
- the cost-versus-risk plot turns the optimization story into something the dashboard can actually show
- the best schedule is only interesting if its sensitivity is visible

## What still feels shaky

- how much the current robustness score should be refined beyond assignment-difference counting
- whether scenario generation should later include probabilistic sampling instead of only fixed multipliers
- how the cost-versus-risk picture will change once forecast uncertainty is introduced more explicitly

## Mistakes or traps

- reporting one optimal schedule without showing how fragile it is
- saving only a picture and not the underlying scenario table
- making sensitivity analysis too abstract to connect to the actual business assumptions

## What exists now

- scheduler sensitivity analysis
- persisted schedule outputs in machine-readable formats
- a cost-versus-risk visualization
- a checkpoint summary artifact that later stages can read directly

## Day 12 conclusion

Day 12 is complete.

The scheduler is now more than an optimizer prototype.

It is a reporting-ready optimization layer with saved outputs and robustness analysis.

## Next move

- add experiment tracking across forecasting variants so comparisons are not trapped in local folders
- connect the forecast and scheduler layers through a cleaner checkpoint command
- start preparing a project checkpoint view rather than only standalone scripts
