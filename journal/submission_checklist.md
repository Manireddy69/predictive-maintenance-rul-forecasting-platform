# Submission Checklist

This file is here to stop end-of-project panic.

The LogicVeda PDF mixes real technical work with capstone presentation requirements.
This checklist keeps both visible.

## Repo sanity

- [x] The repo structure is clear
- [x] Reusable logic lives in `src/`
- [x] Notebooks are for exploration, not the only source of truth
- [x] `.gitignore` is sensible
- [x] `requirements.txt` or equivalent is usable
- [x] Commit messages are readable and meaningful

## Core technical quality

- [x] CMAPSS schema is correct
- [x] Train RUL is derived correctly
- [x] Train/validation split avoids unit leakage
- [x] There is at least one honest baseline before deep learning
- [x] Final metrics are clear
- [x] Error analysis exists

## Documentation quality

- [x] Repo README explains the project at a high level
- [x] Journal notes explain actual reasoning
- [x] There is a clear description of data, preprocessing, modeling, and evaluation
- [x] Important plots are saved or reproducible
- [x] Final report draft exists at `reports/final_project_report_draft.md`
- [x] Demo video script exists at `journal/demo_video_script.md`
- [x] Screenshot checklist exists at `journal/screenshot_checklist.md`
- [x] Compliance report exists at `reports/compliance_report.md`
- [x] Verification metrics exist in `reports/anomaly_acceptance_metrics.csv`, `reports/rul_acceptance_metrics.json`, and `reports/latency_benchmark.json`

## Capstone deliverables

- [x] GitHub repo is ready to share
- [x] Project report PDF is planned
- [x] PDF-ready HTML is exported from the draft at `reports/final_project_report.html`
- [x] Project report PDF is exported from the HTML/draft
- [x] Demo or deployed endpoint is planned
- [ ] Hosted demo URL is rechecked before final submission
- [x] Screenshots are collected under `reports/screenshots/`
- [x] Model artifacts are saved or linked properly
- [x] Video plan exists
- [x] Demo storyboard GIF and slide assets are generated under `reports/demo_assets/`
- [ ] Demo video is recorded with voice-over and uploaded
- [x] Local submission package ZIP is generated at `LogicVeda_Project1_Submission_Package.zip`

## Production-style extras

- [x] Basic tests exist for critical code
- [x] Simple app/dashboard path exists
- [x] Deployment path is defined, even if minimal
- [x] Monitoring or drift ideas are at least documented

## Things I should not force too early

- [x] Kafka
- [x] Airflow
- [ ] Kubernetes
- [x] heavy MLOps tooling

Skipping these early is fine if the core ML work is still being built.

## Weekly questions

- [x] Did I improve understanding, or just add files?
- [x] Can I explain the current pipeline from raw data to prediction?
- [x] Do I know what the current bottleneck is?
- [x] Am I solving the real problem, or drifting into infrastructure theater?

## Final pre-submit actions

- [x] Run the focused test suite one last time
- [x] Launch Streamlit locally and confirm the dashboard loads
- [x] Generate anomaly, RUL, and latency evidence reports
- [x] Capture screenshots using `journal/screenshot_checklist.md`
- [x] Export `reports/final_project_report_draft.md` to PDF-ready HTML
- [x] Print/export `reports/final_project_report.html` to PDF
- [ ] Record the walkthrough using `journal/demo_video_script.md`
- [ ] Confirm all public links work from a browser that is not logged in
- [x] Build local submission package ZIP
