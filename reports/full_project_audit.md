# Full End-to-End Project Audit

Project: Predictive Maintenance & Remaining Useful Life Forecasting Platform  
Repository audited: `C:\Users\Manikanth Reddy\Documents\LogicVeda\Project-1`  
Source of truth: `journal/project_pdf_extracted.txt` extracted from `LV_DSML_Project_1.pdf`  
Audit date: 2026-04-26  

## Executive Verdict

This is a real, working capstone repository with meaningful ML code, tests, CMAPSS data handling, anomaly detection experiments, a maintenance scheduler, Streamlit dashboard assets, a generated PDF report, deployment configuration files, and a static public report path.

It is not yet a production-grade predictive maintenance platform as described by the PDF. The biggest submission risks are:

1. The RUL model misses the PDF quality target: current MAPE is 21.94%, while the PDF asks for 10-12% or better.
2. There is no verified public Streamlit/API live app endpoint. A static report is public, but the interactive app is not.
3. Security, RBAC, TLS, audit trail, drift monitoring, auto-retraining, high availability, and Kubernetes autoscaling are mostly documented or configured at a surface level, not implemented end to end.
4. Anomaly detection evidence is strong for a capstone, but it is largely based on injected/synthetic anomaly labels, not real industrial failure labels.
5. The report and README are slightly ahead of the code in several places, which judges may read as inflated production claims.

Bottom line: strong portfolio prototype, not production hardened. Submit-ready only if framed honestly as a prototype with static report + local dashboard. Risky if presented as fully deployed production infrastructure.

## Verification Snapshot

| Check | Result | Evidence |
|---|---:|---|
| Focused pytest suite | PASS | `38 passed, 2 warnings` |
| Anomaly F1 target >= 0.88 | PASS for MAD/z-score/LOF/LSTM AE | `reports/anomaly_acceptance_metrics.csv` |
| Anomaly precision >= 92% | PASS for top models | precision = 1.0 for MAD/z-score/LOF/LSTM AE |
| Anomaly recall >= 85% | PASS for MAD/z-score/LOF/LSTM AE | recall = 0.8605-0.8721 |
| False alarm rate <= 2% | PASS only for LSTM AE | LSTM AE FAR = 0.00645; MAD/z-score/LOF exceed 2% |
| RUL MAPE <= 12% | FAIL | current MAPE = 21.94% |
| Public static report | PASS | RawGitHack/jsDelivr links documented |
| Public app endpoint | FAIL | Render/Railway configs exist, no verified public app URL |
| Docker support | PARTIAL | `Dockerfile`, `deploy/streamlit/Dockerfile`, `render.yaml`, `railway.json` |
| PDF report | PASS | `reports/Predictive_Maintenance_RUL_Report.pdf` |

## Requirement Compliance Table

Status legend: OK = implemented with evidence, WARN = partial/local/weak evidence, FAIL = missing or does not meet target.

| Requirement | Implemented? | Quality (1-10) | Evidence in Code / Artifacts | Missing / Weakness |
|---|---:|---:|---|---|
| Business impact targets: downtime reduction, maintenance savings, MTBF, ROI | WARN | 5 | `reports/final_project_report_draft.md`, `README.md` | Claimed as expected business value, not validated through real operations data or simulation ROI study. |
| NFR latency: <8s stream-to-alert | FAIL | 2 | `reports/latency_benchmark.json` | Benchmark does not prove Kafka-to-dashboard alert latency. No end-to-end stream alert measurement. |
| NFR batch inference: <300ms | WARN | 5 | `reports/latency_benchmark.json` | Local artifact/scheduler timing only. No real API inference benchmark. |
| NFR throughput: 2-5M sensor readings/day, 100-500 concurrent requests | FAIL | 1 | No load test artifact found | No Locust/k6 results, no concurrency benchmark, no Kafka consumer lag proof. |
| NFR anomaly quality: F1 >= 0.88 | OK | 8 | `reports/anomaly_acceptance_metrics.csv` | Synthetic/injected labels limit realism. Isolation Forest alone fails, but ensemble candidates pass. |
| NFR RUL quality: MAPE <= 12% | FAIL | 4 | `reports/rul_acceptance_metrics.json` | MAPE = 21.94%. This is a major grading risk. |
| NFR availability: 99.7%, retries, dead-letter queues | FAIL | 2 | Kafka consumer modules, Docker configs | No measured availability, retry policy proof, or DLQ integration test. |
| NFR security: JWT, RBAC, TLS, secret rotation, audit logging, sanitization | FAIL | 2 | Security described in report | No implemented auth/RBAC, no enforced TLS, no secret rotation, no audit store. |
| NFR observability: tracing, drift monitoring, SLA alerts | WARN | 4 | `src/mlflow_tracking.py`, report mentions drift | MLflow exists; no request tracing, no Evidently report artifact, no SLA alerting. |
| NFR scalability: Kubernetes HPA, GPU support | WARN | 4 | PyTorch Lightning code, Docker configs | No Kubernetes manifests/HPA. GPU support is framework-capable, not deployed. |
| F-01 multi-protocol real-time and batch ingestion | WARN | 6 | `src/stream_sensor_producer.py`, `src/kafka_to_timescaledb_consumer.py`, `src/batch_pipeline.py`, `dags/batch_csv_to_timescaledb.py` | Good prototype. Does not prove 99.99% delivery, <2s latency, robust DLQ handling. |
| F-02 120+ automated time-series features | WARN | 6 | `src/feature_engineering.py`, `Data/batch/staging/engineered_sensor_features.csv` | Feature pipeline exists, but not clearly versioned, not tracked with feature importance in MLflow, and not integrated into the RUL training path. |
| F-03 multivariate anomaly ensemble | WARN | 7 | `src/anomaly_baseline.py`, `src/anomaly_lstm_autoencoder.py`, `reports/anomaly_acceptance_metrics.csv` | Strong prototype. Synthetic/injected anomaly evidence; Isolation Forest is weak; FAR target only satisfied by LSTM AE. |
| F-04 RUL forecasting with LSTM + TFT + conformal prediction | FAIL | 4 | `src/sequence_attention_model.py`, `src/sequence_data.py`, `reports/rul_acceptance_metrics.json` | LSTM/attention exists. No Temporal Fusion Transformer, no conformal prediction, no calibrated 90% intervals, and target MAPE fails. |
| F-05 cost-optimized maintenance scheduler | OK | 8 | `src/maintenance_scheduler.py`, scheduler tests, schedule artifacts | Good capstone-level MILP. Industrial cost assumptions still need stronger validation. |
| F-06 role-based dashboard and alerting | WARN | 5 | `app/streamlit_app.py`, `app/pages/*`, screenshots | Dashboard exists locally. No roles, no real multi-channel alerts, no audit trail, no proven <=20s refresh SLA. |
| F-07 drift monitoring and auto-retraining | FAIL | 2 | Report/journal mentions only | No working drift report, retraining DAG, A/B promotion, or rollback artifact. |
| Week 1 data/anomaly work | OK | 8 | EDA notebooks, Kafka/batch modules, anomaly scripts, tests | Good enough for checkpoint. Timescale/Kafka are prototype-level unless services are running in demo. |
| Week 2 RUL/scheduler work | WARN | 6 | sequence model, Prophet worker, scheduler, MLflow smoke tests | Scheduler is stronger than RUL. RUL target fails and Prophet/hybrid evidence is inconsistent. |
| Week 3 dashboard/monitoring work | WARN | 5 | Streamlit app and screenshots | Dashboard exists; monitoring/retraining is mostly missing. |
| Week 4 cloud deployment/polish | WARN | 4 | Docker, Render, Railway, GitHub Pages workflow, ZIP/report | Static report is live; no verified public app; no CI test workflow; no load test. |
| PDF report, 10-18 pages | OK | 8 | `reports/Predictive_Maintenance_RUL_Report.pdf` | Must ensure final file size <= 12 MB and claims match reality. |
| Live demo URL or endpoint | WARN | 3 | `reports/deployment_links.md` | Static report URL exists. Interactive app endpoint is missing. This is high risk because live demo is mandatory. |
| GitHub repository | OK | 8 | Git repo and public deployment workflow | Good structure, but needs final README consistency and maybe CI tests. |
| Demo video, 4-8 minutes | WARN | 4 | `journal/demo_video_script.md`, GIF storyboard | Script/assets exist. No confirmed uploaded video URL. |
| Source, notebooks, environment, artifacts | OK | 8 | `src/`, `notebooks/`, `requirements.txt`, `Data/experiments/*` | Requirements are broad/unpinned. Model artifacts are present but heavy. |
| Documentation section 1: hero/cover | OK | 8 | PDF report | Present. |
| Documentation section 2: executive summary | OK | 8 | PDF report | Present. |
| Documentation section 3: business case/objectives | OK | 8 | PDF report | Present. |
| Documentation section 4: functional requirements | OK | 7 | PDF report, compliance report | Present, but statuses should be less optimistic. |
| Documentation section 5: technology stack | OK | 8 | PDF report | Present. |
| Documentation section 6: architecture overview | WARN | 6 | PDF/report visuals | Present, but should distinguish implemented vs target architecture. |
| Documentation section 7: execution timeline | OK | 8 | `journal/day_*.md`, report | Present. |
| Documentation section 8: model performance/evaluation | WARN | 6 | reports/metrics files | Present, but RUL failure and missing NASA score/calibration need clear disclosure. |
| Documentation section 9: MLOps/production readiness | WARN | 5 | MLflow smoke, Docker, configs | Mostly prototype-level. |
| Documentation section 10: challenges/learnings | OK | 7 | report/journal | Present. |
| Documentation section 11: security/privacy/ethics | WARN | 4 | report | Described but not implemented. |
| Documentation section 12: visuals/screenshots | OK | 8 | `reports/screenshots/*`, GIF | Present. Prefer true live app screenshots for final. |
| Documentation section 13: personal reflection | OK | 7 | report | Present. |

## Task 2: Verification Checklist

| Requirement Area | Verification Command / Test |
|---|---|
| Extracted PDF source | `Get-Content -Raw journal/project_pdf_extracted.txt` |
| CMAPSS files exist | `Get-ChildItem Data/CMaps -Filter *.txt` |
| RUL label/window correctness | `python -m pytest tests/test_sequence_data.py -q` |
| Feature engineering pipeline | `python -m pytest tests/test_feature_engineering.py tests/test_ingest_feature_timescaledb.py -q` |
| Batch validation | `python -m pytest tests/test_batch_validation.py -q` |
| Kafka consumer prototype | `python -m pytest tests/test_kafka_to_timescaledb_consumer.py -q` |
| Anomaly baseline code | `python -m pytest tests/test_anomaly_baseline.py -q` |
| LSTM autoencoder anomaly code | `python -m pytest tests/test_anomaly_lstm_autoencoder.py -q` |
| Anomaly acceptance metrics | `Get-Content reports/anomaly_acceptance_metrics.csv` |
| RUL sequence model code | `python -m pytest tests/test_sequence_attention_model.py -q` |
| RUL acceptance metrics | `Get-Content reports/rul_acceptance_metrics.json` |
| Maintenance scheduler | `python -m pytest tests/test_maintenance_scheduler.py -q` |
| MLflow smoke tracking | `python -m pytest tests/test_mlflow_tracking.py -q` |
| Latency benchmark artifact | `Get-Content reports/latency_benchmark.json` |
| Streamlit app launch | `streamlit run app/streamlit_app.py` |
| Docker build | `docker build -t predictive-maintenance-rul .` |
| Static report exists | `Test-Path reports/final_project_report.html` |
| Final PDF exists | `Test-Path reports/Predictive_Maintenance_RUL_Report.pdf` |
| Deployment configs | `Test-Path Dockerfile; Test-Path render.yaml; Test-Path railway.json` |
| Submission ZIP exists | `Test-Path LogicVeda_Project1_Submission_Package.zip` |
| Public report URL | `Invoke-WebRequest <public-report-url> -UseBasicParsing` |
| Public app URL | Must open Render/Railway/Streamlit URL and verify app loads without login. Currently not satisfied. |

## Task 3: Compliance Report Template

| Section | Requirement | My Current Status | Codex Verification | Action Needed |
|---|---|---|---|---|
| Data | CMAPSS FD001-FD004 present and loaded correctly | Partial/Good | Inspect `src/data.py`; run data tests | Add final note explaining exact subset used for model metrics. |
| Labels | Correct train/test RUL labels | Good | `python -m pytest tests/test_sequence_data.py -q` | Add one diagram/table in report showing RUL formula. |
| Features | 120+ rolling/lag/FFT/correlation features | Partial | Inspect `src/feature_engineering.py`; run feature tests | Connect engineered features to model or clearly state they power batch telemetry/anomaly pipeline. |
| Anomaly | Precision >=92%, recall >=85%, F1 >=0.88 | Good but synthetic | Inspect anomaly metrics CSV | Explain injected anomaly limitation. |
| RUL | MAPE <=10-12% | Failed | Inspect RUL metrics JSON | Must improve model or disclose limitation. |
| Scheduler | MILP schedule under 30s | Good | Scheduler tests + latency benchmark | Include final schedule screenshot/table. |
| Dashboard | Role dashboard + alerts | Partial | Run Streamlit locally | Public app URL and real screenshots needed. |
| Drift | Drift monitoring and retraining | Missing | No passing artifact | Add a lightweight drift report or mark as future work. |
| Security | JWT/RBAC/TLS/audit | Missing | No code evidence | Do not claim implemented. |
| Deployment | Public live demo | Partial/Missing | Static report works; app missing | Deploy Streamlit to Render/Streamlit Cloud/Railway. |
| Documentation | PDF 10-18 pages with required sections | Good | PDF exists | Ensure final report is honest about limitations. |
| Video | 4-8 minute demo | Partial | Script/GIF only | Record and upload final video. |

## Technical Deep Review

### Data

Strengths:

- `src/data.py` uses the standard CMAPSS 26-column layout and correctly assigns operating settings plus sensor columns.
- Training RUL is generated as `max_cycle - cycle`, which is the standard uncapped raw RUL formulation.
- `src/sequence_data.py` correctly joins NASA test RUL values by unit and computes row-level test RUL as total cycles to failure minus current cycle.
- Window generation aligns the label to the end of the window.
- Train/validation split is unit-aware, reducing leakage.
- Scaler fitting is done on training windows only in the sequence pipeline.

Weaknesses:

- No capped RUL target is used or compared. CMAPSS baselines often cap RUL at 125/130 cycles to improve stability.
- FD002 and FD004 have multiple operating conditions, but there is no strong condition-specific normalization or clustering evidence.
- The final RUL evidence appears centered on FD001-style sequence predictions. The PDF context expects robustness across conditions/noise.
- No NASA scoring function or early/late penalty metric is present.

### Feature Engineering

Strengths:

- `src/feature_engineering.py` includes rolling features, lags, FFT amplitudes, and cross-sensor ratios.
- Batch staging artifacts exist under `Data/batch/staging`.

Weaknesses:

- The engineered feature pipeline is not clearly integrated into the RUL model path.
- Feature importance is not tracked in MLflow as required by F-02.
- Health indicators, PCA/time-series embeddings, and degradation trend features are not strongly evidenced.

### Models

Strengths:

- Anomaly detection includes statistical methods, Isolation Forest/LOF, and LSTM autoencoder.
- RUL model includes a BiLSTM/attention sequence model.
- Prophet worker/hybrid components exist.
- Scheduler uses PuLP MILP, which matches the PDF direction.

Weaknesses:

- No Temporal Fusion Transformer implementation.
- No conformal prediction or calibrated 90% prediction intervals.
- No strong tabular RUL baselines such as Random Forest, XGBoost, or LightGBM.
- RUL model misses the target by a large margin.

### Validation

Strengths:

- Unit-wise splitting and tests are a good sign.
- Focused tests pass.
- Metrics artifacts exist and are machine-readable.

Weaknesses:

- No cross-validation or repeated seed robustness evidence.
- No per-engine error distribution report.
- No FD001-FD004 aggregate benchmark table.
- No stress/noise/missing-data robustness report for RUL.

### Metrics

Strengths:

- Anomaly metrics include ROC-AUC, PR-AUC, threshold, precision, recall, F1, FAR, and confusion counts.
- RUL metrics include RMSE, MAE, R2, MAPE, median absolute error, and within-cycle rates.

Weaknesses:

- RUL MAPE fails.
- NASA scoring function is missing.
- Prediction interval calibration is missing.
- Early failure penalty is missing.

### Explainability

Strengths:

- Attention peak metadata exists in sequence outputs.
- Some visuals are included in the report/screenshots.

Weaknesses:

- No SHAP plots.
- No feature importance tracked in MLflow.
- Attention visualization alone is not enough for industrial explainability.

### Engineering

Strengths:

- Code is modular: `src/`, `tests/`, `app/`, `dags/`, `db/`, `deploy/`.
- Focused tests pass.
- Artifacts are organized by reports and experiment folders.
- Docker and hosting config files exist.

Weaknesses:

- `requirements.txt` is broad and mostly unpinned, which hurts reproducibility.
- No full CI test workflow is present; GitHub Actions is used for static report deployment only.
- No single `make` or CLI entrypoint that reproduces all metrics from scratch.
- Some artifacts are generated/demo-oriented and should not be overclaimed as live production evidence.

### Deployment

Strengths:

- Static report can be served from the `gh-pages` branch and alternate CDN links are documented.
- Dockerfile, Render config, Railway config, and Streamlit Dockerfile exist.

Weaknesses:

- No verified public Streamlit, FastAPI, Flask, or model inference endpoint.
- No HTTPS app evidence beyond static report hosting.
- No API endpoint for batch or online prediction.
- No Kubernetes manifests or HPA despite PDF target stack.

## Fake / Weak Work Detection

This does not look like fake work overall. There is real code, real tests, real artifacts, and a coherent project structure.

The weak areas are mostly overclaiming and proof gaps:

- Production language is stronger than implementation evidence.
- Anomaly metrics look good, but they depend on injected/synthetic anomaly labels.
- RUL claims are weak because the main metric fails the PDF threshold.
- Dashboard screenshots and GIFs help presentation, but they are not a substitute for a public live app.
- Static report deployment may be mistaken for app deployment; the submission should distinguish them clearly.
- Security, availability, drift, and autoscaling should be framed as planned architecture unless implemented.
- README/report metrics must be aligned with the latest acceptance files.

## Reviewer Perspectives

### Hiring Manager

Impression: good initiative, broad ML engineering exposure, and a portfolio-friendly story. Concern: the candidate may be claiming production readiness without production proof. Best interview angle is to say, "I built an end-to-end prototype, then audited the gap to production."

### Kaggle Expert

Impression: the RUL modeling is not competitive enough yet. Missing capped RUL, stronger baselines, NASA score, condition-aware normalization, and per-FD evaluation. The RUL MAPE failure will be noticed.

### ML Engineer

Impression: repo organization is solid, but reproducibility and deployment are incomplete. I would ask for a one-command training/evaluation path, pinned dependencies, CI tests, and a real inference endpoint.

### Data Science Professor

Impression: good breadth and documentation. The modeling section needs more scientific discipline: baselines, ablations, uncertainty calibration, and clearer separation between real results and target architecture.

## Final Score

### LogicVeda-Style 100 Point Estimate

| Category | Points | Score | Reason |
|---|---:|---:|---|
| Business Understanding & Impact | 15 | 13 | Clear use case and ROI framing. |
| Technical Depth & Model Quality | 25 | 15 | Good anomaly/scheduler depth, but RUL target fails and TFT/conformal missing. |
| MLOps & Production Readiness | 20 | 9 | Docker/configs/MLflow exist; public app, security, drift, CI, load testing missing. |
| Code Quality & Reproducibility | 15 | 12 | Modular code and tests; dependencies and full reproduction path need work. |
| Documentation & Presentation | 15 | 13 | Strong PDF/screenshots, but must reduce inflated claims. |
| Demo & Video Polish | 10 | 5 | Static report and demo assets exist; public app/video not verified. |
| Total | 100 | 67 | Strong prototype, not fully compliant with the production-grade PDF. |

### Eight-Dimension Reviewer Score

| Dimension | Score / 10 |
|---|---:|
| Business relevance | 8 |
| Data science rigor | 6 |
| ML engineering quality | 7 |
| Modeling quality | 5 |
| Interpretability | 3 |
| Production readiness | 4 |
| Resume value | 7 |
| Originality | 6 |

Overall submission-readiness score: 67/100.

## Fix These First

### Must Fix Before Submission

1. Deploy the Streamlit app publicly and add the URL to `README.md`, `reports/deployment_links.md`, and the PDF/report appendix.
2. Record and upload a 4-8 minute demo video. Include dashboard, anomaly metrics, RUL limitation, scheduler output, and static report.
3. Update README metrics to match `reports/anomaly_acceptance_metrics.csv` and `reports/rul_acceptance_metrics.json`.
4. Be honest about RUL: current MAPE is 21.94%, so either improve it or explicitly label it as below production target.
5. Add a simple NASA score and per-engine RUL error table to strengthen evaluation, even if MAPE remains high.
6. Add a final screenshot from the actual deployed app, not only generated report assets.
7. Mark security, drift, RBAC, HPA, and production SLA items as "future production hardening" unless you implement them.

### Nice To Have

1. Add a tabular RUL baseline using Random Forest or LightGBM/XGBoost with capped RUL.
2. Add SHAP or permutation feature importance for a tabular baseline.
3. Add condition-aware normalization for FD002/FD004.
4. Add `.github/workflows/ci.yml` running focused pytest.
5. Add a compact model card documenting training data, metrics, limitations, and intended use.

### Overkill / Waste of Time Before Tomorrow

1. Full Kubernetes/EKS/GKE deployment.
2. Real Kafka exactly-once production guarantees.
3. Full JWT/RBAC dashboard implementation.
4. Building a Temporal Fusion Transformer from scratch.
5. Complete Prometheus/Grafana/Sentry stack.

### How To Make This Look Like A 2 Years Experience Project

1. Add a one-command reproducibility path: `python -m src.evaluate_all --quick`.
2. Add CI that runs the focused tests on every push.
3. Add a public Streamlit demo plus a static report fallback.
4. Add a model card and an audit section that clearly separates implemented, prototype, and future production architecture.
5. Add one strong baseline/ablation table: naive last-cycle, Random Forest/LightGBM, LSTM attention.
6. Add real limitations: "RUL does not meet target yet; next iteration uses capped RUL, condition normalization, and NASA score optimization."

## Patch-Ready Recommendations

### 1. CI Workflow

Recommended file: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run focused tests
        run: |
          python -m pytest \
            tests/test_anomaly_baseline.py \
            tests/test_anomaly_lstm_autoencoder.py \
            tests/test_kafka_to_timescaledb_consumer.py \
            tests/test_sequence_data.py \
            tests/test_sequence_attention_model.py \
            tests/test_maintenance_scheduler.py \
            tests/test_mlflow_tracking.py -q
```

### 2. NASA RUL Score Helper

Recommended file: `src/rul_scoring.py`

```python
from __future__ import annotations

import numpy as np


def nasa_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA CMAPSS asymmetric score. Late predictions are penalized more."""
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = errors < 0
    scores = np.empty_like(errors, dtype=float)
    scores[early] = np.exp(-errors[early] / 13.0) - 1.0
    scores[~early] = np.exp(errors[~early] / 10.0) - 1.0
    return float(np.sum(scores))


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
```

### 3. README Honesty Block

Add this near the metrics section:

```markdown
### Current Acceptance Status

- Anomaly detection meets the capstone F1/precision/recall target on the injected-anomaly benchmark.
- LSTM autoencoder is the only anomaly model currently below the 2% false-alarm target.
- RUL forecasting is implemented, but the current MAPE is 21.94%, which does not meet the 10-12% production target from the specification.
- The repository includes Docker/Render/Railway deployment configuration, but the public interactive app URL must be verified separately from the static report link.
```

## Final Submission Risk

High risk if submitted as "fully production-grade."  
Moderate risk if submitted as "working end-to-end prototype with honest production hardening roadmap."  
Low risk for code structure/documentation if the final README, video, and demo links are cleaned up today.

