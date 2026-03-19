#!/bin/bash
# =============================================================================
# Longleaf SLURM script — ABCD6 Canonical Pipeline Run
# Runs all 4 analysis notebooks in sequence with 1000 perms / 10000 bootstraps
#
# Usage:
#   sbatch slurm/run_canonical.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/canonical_run_<JOBID>.out
# =============================================================================

#SBATCH --job-name=abcd6_canonical
#SBATCH --output=logs/canonical_run_%j.out
#SBATCH --error=logs/canonical_run_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_ONYEN@ad.unc.edu

# ---- Environment ----
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module purge
module load anaconda3/2023.03

# Activate conda env (create it first with: conda env create -f environment.yml)
# Or: conda create -n abcd6 python=3.9 && conda activate abcd6 && pip install -r requirements.txt
conda activate abcd6

# ---- Directory ----
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
mkdir -p logs

# ---- Run notebooks in order ----
echo ""
echo "========================================="
echo "STEP 1/4: Data preparation (00_dataprep)"
echo "========================================="
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/00_dataprep.ipynb
echo "00_dataprep complete: $(date)"

echo ""
echo "============================================="
echo "STEP 2/4: Cross-sectional (01_crosssectional)"
echo "============================================="
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=36000 \
    notebooks/01_crosssectional.ipynb
echo "01_crosssectional complete: $(date)"

echo ""
echo "======================================="
echo "STEP 3/4: Prospective (02_prospective)"
echo "======================================="
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=36000 \
    notebooks/02_prospective.ipynb
echo "02_prospective complete: $(date)"

echo ""
echo "=========================================="
echo "STEP 4/4: Change score (03_change_score)"
echo "=========================================="
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=36000 \
    notebooks/03_change_score.ipynb
echo "03_change_score complete: $(date)"

echo ""
echo "========================================="
echo "ALL STEPS COMPLETE: $(date)"
echo "========================================="
echo "Outputs written to: outputs/regression/"
echo "Run ID is in configs/run.yaml"
