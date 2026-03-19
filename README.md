# ABCD Release 6 — Dopaminergic Asymmetry & Psychosis-Spectrum Regression

**Honors Thesis | University of North Carolina at Chapel Hill | Department of Computer Science | Spring 2026**

This repository contains the primary analysis pipeline for:

> *Dopaminergic Subcortical Brain Asymmetry and Psychosis-Spectrum Symptoms in the ABCD Study: A Machine Learning Analysis*

Using ABCD Study Release 6.0, we test whether dopaminergic subcortical brain asymmetry predicts psychosis-spectrum symptom severity (PQ-BC) in a community sample of 9–10 year-old children, and whether this association holds prospectively or reflects a concurrent state marker.

---

## Key Results

| Aim | Analysis | N | r | p |
|-----|----------|---|---|---|
| 1 (Cross-sectional) | SVR: Subcortical dopamine AI → baseline PPS | 474 | **0.164** | < 0.001 |
| 1 (Pallidum univariate) | Pallidum AI → PPS (Bonferroni) | 474 | **−0.154** | 0.005 |
| 2 (Prospective Y2) | SVR: Baseline brain → Y2 PPS | 209 | −0.096 | ns |
| 2 (Prospective Y4) | SVR: Baseline brain → Y4 PPS | 137 | −0.011 | ns |
| 3 (Change score) | SVR: ΔBrain → ΔPPS | 7,276 | 0.001 | ns |

---

## Repository Structure

```
abcd6/
├── notebooks/          # Numbered analysis notebooks (run in order)
│   ├── 00_dataprep.ipynb         # Data loading, QC, train/test splits
│   ├── 01_crosssectional.ipynb   # Aim 1: cross-sectional SVR + robustness
│   ├── 02_prospective.ipynb      # Aim 2: prospective prediction + state marker
│   └── 03_change_score.ipynb     # Aim 3: change-score prediction
├── src/core/           # Modular source code
│   ├── config.py               # Config loader, run ID generator
│   ├── features.py             # ROI feature selection
│   ├── preprocessing/          # QC, missing data, splits, ICV correction
│   ├── harmonize/              # ComBat harmonization (neuroHarmonize)
│   └── regression/             # SVR pipeline, evaluation, robustness
├── configs/            # YAML configuration files
│   ├── data.yaml               # Data paths, QC thresholds, ROI definitions
│   ├── regression.yaml         # Model params, n_permutations=1000, n_bootstrap=10000
│   ├── harmonize.yaml          # ComBat settings
│   └── run.yaml                # Seed, run name
├── outputs/            # Run outputs (gitignored — too large)
└── requirements.txt    # Pinned Python dependencies
```

---

## Methodological Design

- **Model**: Support Vector Regression (linear kernel, C=1.0, ε=0.1)
- **Cross-validation**: 5-fold `StratifiedGroupKFold` with family ID grouping (zero sibling leakage)
- **Harmonization**: ComBat (neuroHarmonize) per-fold — fit on training, applied to test
- **ICV correction**: Ratio-based (V/ICV, CT/ICV^1/3) before ComBat
- **Residualization**: Age + sex residualized from target per-fold on training data only
- **Permutation testing**: 1,000 pipeline-matched permutations
- **Bootstrap CIs**: 10,000 resamples (percentile method)
- **Asymmetry Index**: AI = (L−R)/(L+R), ENIGMA convention

---

## Replication

### Requirements

```bash
conda create -n abcd6 python=3.9
conda activate abcd6
pip install -r requirements.txt
```

### Data

ABCD Release 6.0 data requires a Data Use Agreement with the [NIMH Data Archive](https://nda.nih.gov/). Place parquet files in `data/raw/` as specified in `configs/data.yaml`.

### Running the Pipeline

Run notebooks in order:

```bash
jupyter nbconvert --to notebook --execute notebooks/00_dataprep.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_prospective.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_change_score.ipynb
```

Or use the Longleaf SLURM script: `slurm/run_canonical.sh`

---
