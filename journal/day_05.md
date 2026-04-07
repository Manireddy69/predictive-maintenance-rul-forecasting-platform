# Day 05

Today was about earning the right to do deep learning later.

It is very easy in predictive maintenance to jump straight to LSTM, autoencoder, transformer, or whatever sounds advanced that week.

That is not the same thing as doing honest work.

Day 5 was the day to build baseline anomaly detectors that are simple enough to debug and strong enough to be respected.

## What I worked on

- built a baseline anomaly detection experiment
- trained on normal-only data
- injected controlled anomalies into a hold-out set
- evaluated:
  - `Isolation Forest`
  - `Local Outlier Factor`
- measured performance using:
  - ROC-AUC
  - PR-AUC
  - precision-recall curves

## Why I trained only on normal data

That choice matters more than it first sounds.

In anomaly detection, the practical setup is often:

- a lot of normal data
- very limited reliable anomaly labels
- anomalies that are diverse rather than one clean class

So I treated the problem as mostly unsupervised or one-class style detection.

Training on normal-only data made sense because I wanted the detector to learn what normal behavior looks like and then score deviations from that.

That is closer to the real anomaly problem than pretending I have a perfect balanced classification dataset.

## Why I injected anomalies into the hold-out set

I needed a controlled test.

If I only train and evaluate on whatever happened to be in the synthetic set, it becomes harder to understand what the detector is responding to.

By injecting anomalies into the hold-out set, I could:

- preserve a normal-only training distribution
- create known abnormal cases
- measure the detector with actual labels during evaluation

This is not meant to prove production readiness.
It is meant to make the comparison disciplined.

## Why I chose `Isolation Forest`

`Isolation Forest` is a useful baseline because it is conceptually simple in a good way.

It works by isolating unusual points more quickly in random partitioning trees.

Why I used it:

- it is a standard anomaly baseline
- it does not need labels
- it handles higher-dimensional input reasonably well
- it gives a useful "global oddness" score

Why I did not treat it as automatically strong:

it can be useful, but it is still a generic detector.
If the anomaly structure is mostly local or density-based, it may not be the best match.

## Why I chose `Local Outlier Factor`

`LOF` asks a different question from Isolation Forest.

It focuses on local density.
In plain language:
does this point look sparse compared with its neighbors?

Why I used it:

- it gives a different kind of anomaly logic than tree isolation
- it is often good when anomalies are unusual relative to nearby normal clusters
- it is a strong complement to Isolation Forest because the methods fail in different ways

That difference was important.
I did not want two baselines that were basically the same idea wearing different names.

## How I approached the comparison

I wanted the experiment to stay fair.

So the shape was:

1. train on normal-only data
2. create a hold-out split
3. inject controlled anomalies into the hold-out set
4. score the hold-out rows
5. compare metrics using the same evaluation setup

That mattered because anomaly comparisons are easy to fake if the train and evaluation story is muddy.

## Why I used ROC-AUC and PR-AUC

I did not want to rely on just one metric.

### ROC-AUC

Useful because it tells me how well the detector separates normal from anomaly across thresholds.

But it can look more optimistic than it should when anomalies are relatively rare.

### PR-AUC

This mattered more for anomaly detection because the positive class is the abnormal class.
Precision-recall behavior is often more informative when class imbalance is part of the problem.

That is why I kept both metrics and the precision-recall curve itself.

## What the results said

On the row-level synthetic experiment:

- `Isolation Forest`: ROC-AUC `0.6398`, PR-AUC `0.1886`
- `Local Outlier Factor`: ROC-AUC `0.7335`, PR-AUC `0.5060`

What that meant in plain language:

- `LOF` was clearly stronger than `Isolation Forest` in this setup
- the anomaly structure I created seems to favor local-density reasoning more than tree-based isolation
- `Isolation Forest` was useful as a baseline, but not the strongest candidate

That was good to learn.
It gave the project a real baseline instead of a ceremonial one.

## What changed in my head

Before running the comparison, it was easy to think:
"these are just warm-up methods before the real model."

That was the wrong mindset.

After seeing the results, the better mindset was:

- a baseline is not there to be weak
- a baseline is there to force future models to justify themselves
- if the simple detector is strong, that is useful information

That changed how I thought about Day 6.

## Why this matters for the next algorithm choice

The Day 5 result set up the Day 6 question properly.

The question was no longer:
"can I build an LSTM autoencoder?"

The better question was:
"can the LSTM autoencoder beat a baseline that already works reasonably well?"

That is a much better reason to move into deep learning.

## What still felt shaky

- how much the injected anomaly design was shaping the detector ranking
- whether row-level scoring was enough for a sequential maintenance problem
- whether the strongest anomalies were too obvious compared to what real systems would show

## Mistakes I wanted to avoid

- training on mixed normal and anomaly data and pretending it was a clean anomaly setup
- using only ROC-AUC and missing the class-imbalance story
- treating the weaker baseline as "good enough" just so the deep model could look better later
- jumping into LSTM before understanding what the baseline already solved

## What I am taking from Day 5

Day 5 gave the anomaly track its first honest benchmark.

That matters because deep learning should come after strong baselines, not instead of them.

## Next move

Now that the baseline existed, the next step was:

- move into sequence-aware anomaly detection
- use reconstruction error instead of just static outlier scoring
- compare the deep model against methods that already earned respect
