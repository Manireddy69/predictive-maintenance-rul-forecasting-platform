# Submission Checklist

This file is here to stop end-of-project panic.

The LogicVeda PDF mixes real technical work with capstone presentation requirements.
This checklist keeps both visible.

## Repo sanity

- [ ] The repo structure is clear
- [ ] Reusable logic lives in `src/`
- [ ] Notebooks are for exploration, not the only source of truth
- [ ] `.gitignore` is sensible
- [ ] `requirements.txt` or equivalent is usable
- [ ] Commit messages are readable and meaningful

## Core technical quality

- [ ] CMAPSS schema is correct
- [ ] Train RUL is derived correctly
- [ ] Train/validation split avoids unit leakage
- [ ] There is at least one honest baseline before deep learning
- [ ] Final metrics are clear
- [ ] Error analysis exists

## Documentation quality

- [ ] Repo README explains the project at a high level
- [ ] Journal notes explain actual reasoning
- [ ] There is a clear description of data, preprocessing, modeling, and evaluation
- [ ] Important plots are saved or reproducible

## Capstone deliverables

- [ ] GitHub repo is ready to share
- [ ] Project report PDF is planned
- [ ] Demo or deployed endpoint is planned
- [ ] Screenshots are being collected
- [ ] Model artifacts are saved or linked properly
- [ ] Video plan exists

## Production-style extras

- [ ] Basic tests exist for critical code
- [ ] Simple app/dashboard path exists
- [ ] Deployment path is defined, even if minimal
- [ ] Monitoring or drift ideas are at least documented

## Things I should not force too early

- [ ] Kafka
- [ ] Airflow
- [ ] Kubernetes
- [ ] heavy MLOps tooling

Skipping these early is fine if the core ML work is still being built.

## Weekly questions

- [ ] Did I improve understanding, or just add files?
- [ ] Can I explain the current pipeline from raw data to prediction?
- [ ] Do I know what the current bottleneck is?
- [ ] Am I solving the real problem, or drifting into infrastructure theater?
