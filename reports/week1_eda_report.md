# Week 1 EDA Report

## Scope

This report summarizes the first-pass exploratory analysis for the `FD001` split of the NASA CMAPSS dataset.
The goal was to understand the shape of the sequence data before leaning too hard on modeling decisions.

## Dataset Summary

- Train units: `100`
- Test units: `100`
- Train cycle range: `128` to `362`
- Test cycle range: `31` to `303`
- Test RUL rows: `100`
- Train rows: `20631`
- Test rows: `13096`

## What was checked

- distribution spread and sensor variability
- constant or near-dead sensors
- rough degradation signal from start-to-end sensor drift
- missing-value patterns
- stationarity behavior using ADF and KPSS
- FFT-based frequency signatures for high-variability sensors

## Constant Columns

setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19

## Most Variable Sensors

| sensor | std | nunique |
| --- | --- | --- |
| sensor_9 | 22.0829 | 6403.0 |
| sensor_14 | 19.0762 | 6078.0 |
| sensor_4 | 9.0006 | 4051.0 |
| sensor_3 | 6.1311 | 3012.0 |
| sensor_17 | 1.5488 | 13.0 |
| sensor_7 | 0.8851 | 513.0 |
| sensor_12 | 0.7376 | 427.0 |
| sensor_2 | 0.5001 | 310.0 |

## Strongest Average Degradation Deltas

| sensor | avg_end_minus_start |
| --- | --- |
| sensor_9 | 43.5201 |
| sensor_14 | 30.6965 |
| sensor_4 | 28.4353 |
| sensor_3 | 16.0833 |
| sensor_17 | 4.5 |
| sensor_7 | -2.6344 |
| sensor_12 | -2.2279 |
| sensor_2 | 1.3246 |

## Missing Values

| column | missing_count |
| --- | --- |
| none | 0 |

## Stationarity Snapshot

ADF and KPSS were used together so the report would not lean too heavily on one stationarity test alone.

| sensor | adf_pvalue | kpss_pvalue | adf_stationary | kpss_stationary |
| --- | --- | --- | --- | --- |
| sensor_9 | 0.0 | 0.1 | True | True |
| sensor_14 | 0.0 | 0.1 | True | True |
| sensor_4 | 0.0 | 0.1 | True | True |
| sensor_3 | 0.0 | 0.1 | True | True |

## FFT Snapshot

The FFT table captures dominant amplitude strengths for the most variable sensors and helps justify frequency-aware features later in the pipeline.

| sensor | fft_amp_1 | fft_amp_2 | fft_amp_3 | fft_amp_4 | fft_amp_5 |
| --- | --- | --- | --- | --- | --- |
| sensor_9 | 53619.2854 | 45085.8048 | 44420.7116 | 39949.5778 | 39004.1203 |
| sensor_14 | 44372.4739 | 44346.0259 | 37534.3288 | 36577.8641 | 35293.5194 |
| sensor_4 | 25708.8631 | 16863.1175 | 16093.8391 | 15691.8296 | 14980.5642 |
| sensor_3 | 15856.6264 | 9955.6281 | 9686.4256 | 9552.1937 | 9547.0317 |

## Initial Observations

- The dataset is sequential and unit-based, not ordinary tabular regression data.
- Sensor behavior is uneven: some sensors move a lot, others are much less informative.
- The start-vs-end drift view suggests that a subset of sensors carries stronger degradation signal than the rest.
- Stationarity should not be assumed blindly; some high-variability sensors show evidence of evolving behavior over time.
- Frequency information is present strongly enough to justify FFT-derived features in the feature engineering pipeline.

## Why this mattered for Week 1

- It justified keeping CMAPSS as the main sequence-learning dataset.
- It supported the later choice to add rolling, lag, and FFT features.
- It reinforced that anomaly experiments should not be treated as simple row classification.
- It gave a cleaner foundation before moving into anomaly baselines and sequence models.

## Sample Derived RUL View

| unit | cycle | rul |
| --- | --- | --- |
| 1 | 1 | 191 |
| 1 | 2 | 190 |
| 1 | 3 | 189 |
| 1 | 4 | 188 |
| 1 | 5 | 187 |
| 1 | 6 | 186 |
| 1 | 7 | 185 |
| 1 | 8 | 184 |
