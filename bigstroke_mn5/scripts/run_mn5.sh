#!/usr/bin/env bash
#SBATCH --job-name=bigstroke_m1
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --time=08:00:00
#SBATCH --qos=gp_bsc            # or appropriate queue at submission time
#SBATCH --account=YOURACCOUNT   # replace with allocation account

# MareNostrum 5 SLURM submission for full-scale M1 microcircuit run.
# Adapt --account, --qos, --time as required by your RES allocation.
#
# Submit with:   sbatch scripts/run_mn5.sh

set -euo pipefail

# ── Software modules ─────────────────────────────────────────────────
module purge
module load gcc openmpi nest python      # adapt to actual MN5 module names

# ── Project setup ────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
mkdir -p output

# Each SLURM task is one MPI rank; OMP_NUM_THREADS = --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Job   : $SLURM_JOB_ID"
echo "Nodes : $SLURM_JOB_NUM_NODES"
echo "Ranks : $SLURM_NTASKS"
echo "OMP/T : $OMP_NUM_THREADS"

# ── Run full-scale simulation ────────────────────────────────────────
# srun handles MPI launch; NEST takes care of distribution across ranks
srun python -m bigstroke_mn5.m1_microcircuit.simulate \
    --scale 1.0 \
    --stroke-fraction 0.0 \
    --sim-time-ms 5000 \
    --label mn5_full_healthy \
    --threads "$OMP_NUM_THREADS"

srun python -m bigstroke_mn5.m1_microcircuit.simulate \
    --scale 1.0 \
    --stroke-fraction 0.5 \
    --sim-time-ms 5000 \
    --label mn5_full_stroke \
    --threads "$OMP_NUM_THREADS"

echo "Done."
