# Day 06 Anomaly Detection Summary

- Run name: `smoke_save_check`
- Best model: `MAD Distance`
- Best PR-AUC: `0.939986`
- Reconstruction-error threshold: `1.270194`
- Training device: `cpu`
- Final training loss: `0.942150`

## Comparison Table

| model_key | model_name | roc_auc | pr_auc | pr_curve_points |
| --- | --- | --- | --- | --- |
| mad | MAD Distance | 0.941561 | 0.939986 | 113 |
| zscore | Z-Score Distance | 0.940698 | 0.939898 | 116 |
| local_outlier_factor | Local Outlier Factor | 0.918339 | 0.921313 | 123 |
| lstm_autoencoder | LSTM Autoencoder | 0.923389 | 0.908206 | 262 |
| isolation_forest | Isolation Forest | 0.759402 | 0.672790 | 115 |

## Plain-Language Takeaway

The LSTM autoencoder learned useful temporal structure, but the simplest robust statistical method in this run was `MAD Distance`.
That means deep learning is competitive here, but not automatically better than well-chosen statistical baselines.
