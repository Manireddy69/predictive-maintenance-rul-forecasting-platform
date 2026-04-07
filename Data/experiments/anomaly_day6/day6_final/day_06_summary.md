# Day 06 Anomaly Detection Summary

- Run name: `day6_final`
- Best model: `LSTM Autoencoder`
- Best PR-AUC: `0.966741`
- Reconstruction-error threshold: `0.120010`
- Training device: `cpu`
- Final training loss: `0.096421`

## Comparison Table

| model_key | model_name | roc_auc | pr_auc | pr_curve_points |
| --- | --- | --- | --- | --- |
| lstm_autoencoder | LSTM Autoencoder | 0.966992 | 0.966741 | 242 |
| mad | MAD Distance | 0.948537 | 0.952635 | 83 |
| zscore | Z-Score Distance | 0.944336 | 0.948949 | 85 |
| local_outlier_factor | Local Outlier Factor | 0.926444 | 0.936447 | 91 |
| isolation_forest | Isolation Forest | 0.764216 | 0.709978 | 82 |

## Plain-Language Takeaway

The LSTM autoencoder learned useful temporal structure, but the simplest robust statistical method in this run was `LSTM Autoencoder`.
That means deep learning is competitive here, but not automatically better than well-chosen statistical baselines.
