# Day 01

Today was basically about slowing down and not pretending I understand the dataset just because I can load it.

The temptation here is obvious:
load the files, print shapes, make a few plots, and then move on to modeling.

That would be weak.

What I actually needed to understand first was what CMAPSS is really giving me.

## What I figured out

CMAPSS is not normal tabular data.
Each `unit` is one engine.
Each `cycle` is one step in that engine's life.

The train set shows the engine all the way until failure.
The test set does not.
It stops earlier, and the separate RUL file tells me how much life was left after the last observed test cycle.

That point matters more than most of the Day 1 plots.
If I do not understand this, the rest of the project becomes performative.

## What I worked on

- loaded `train_FD001.txt`
- loaded `test_FD001.txt`
- loaded `RUL_FD001.txt`
- checked the CMAPSS schema properly
- derived the train RUL target
- looked at which sensors seem useful and which ones are basically dead weight

## What changed in my head

Before this, it was easy to think:
"this is just a dataset with sensor columns and a target I need to predict."

That is not really the problem.

The actual problem is:
- the data is sequential
- the target has to be constructed for train
- the test setup is different from the train setup
- the split strategy later has to respect unit boundaries

That is a much more specific problem, and honestly a more interesting one.

## What seems important now

- getting the schema right
- understanding how train RUL is created
- not using random row splits later
- identifying constant sensors early so I do not carry garbage into the model

## What felt easy to fake

A lot of Day 1 work can look productive without being useful.

Examples:
- too many summary tables
- too many plots with no conclusion
- talking about "EDA" without making any decisions

That is the trap I want to avoid.

## What still feels uncertain

- which exact sensor subset will be strongest for FD001
- whether capped RUL will help or just simplify the target artificially
- how far simple baseline models can go before sequence models are actually justified

## Mistakes I want to avoid next

- treating rows like independent samples
- building features before I lock down the split logic
- jumping into LSTM because it sounds like the expected next step
- mixing anomaly detection and RUL forecasting into one vague story

## What I am taking from Day 1

Day 1 is not about “doing EDA.”
It is about making sure the project is real.

If the target logic, schema, and split logic are wrong, later work might still look advanced, but it will not be trustworthy.

## Next move

Next I need to make the preprocessing and split logic solid.
That matters more than model complexity right now.
