# UToE 2.1 — Logistic–Scalar Adversarial Test (Preregistered)

This repository contains analytic code implementing the preregistered adversarial model-comparison pipeline for the Unified Theory of Emergence (UToE 2.1).

## Preregistration (OSF)
- DOI: 10.17605/OSF.IO/HV7S5

## Core model

dΦ/dt = r · λ · γ · Φ · (1 − Φ / Φ_max)

K(t) = λ · γ · Φ(t)

Where:
- Φ(t): scalar integration trajectory derived from an empirical time series R(t)
- Φ_max: saturation bound
- r: temporal scaling constant
- λ: coupling strength proxy
- γ: coherence efficiency proxy
- K(t): structural intensity diagnostic

## Confirmatory comparison
Primary confirmatory comparison is:
- M1: Logistic–scalar model
- M2: Linear growth baseline

Decision criteria (preregistered):
- ΔAIC = AIC(M1) − AIC(M2) ≤ −5
- ΔAdjR² = AdjR²(M1) − AdjR²(M2) ≥ 0.01

Robustness control (preregistered):
- Phase-randomized surrogate of R(t); logistic–scalar should degrade under phase randomization.

## Data provenance
External datasets referenced by the preregistration:
- EEGBCI / PhysioNet (DOI: 10.13026/C28G6P)
- OpenNeuro ds005620 (DOI: 10.18112/openneuro.ds005620.v1.0.0)

This repository does not redistribute raw data; it provides deterministic loaders and analysis scripts.

## Repository contents (planned)
- `analysis_pipeline.py`: end-to-end confirmatory pipeline (load → Φ(t) → fit → compare → phase-randomization)
- `requirements.txt`: pinned dependencies
- `LICENSE`: MIT
- `CITATION.cff`: citation metadata (added prior to Zenodo archival)

## Reproducibility
All steps are implemented deterministically with fixed preprocessing and preregistered decision thresholds. Any additional analyses beyond the preregistration are labeled exploratory in separate scripts/notebooks.
