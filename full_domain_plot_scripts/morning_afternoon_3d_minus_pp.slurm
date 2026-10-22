#!/bin/bash
#SBATCH --job-name=HB_SW_AM_PM
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/polar_sw_am_pm-%j.out
#SBATCH --error=logs/polar_sw_am_pm-%j.err

set -euo pipefail

echo "Start: $(date)"
echo "Node:  $SLURM_NODELIST"
echo "Job:   $SLURM_JOB_ID"

module purge
module load slurm/alpine
module load anaconda

export CONDA_PREFIX=/projects/battobrah@xsede.org/software/anaconda/envs/HB3
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PROJ_LIB="$CONDA_PREFIX/share/proj"
export PROJ_DATA="$CONDA_PREFIX/share/proj"
export GDAL_DATA="$CONDA_PREFIX/share/gdal"

# Each worker is single-threaded.
# Python parallelizes across catchment IDs (CIDs).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev

SCRIPT="$REPO/full_domain_plot_scripts/morning_afternoon_3d_minus_pp.py"

SIMS=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations

export HB_PATH_3D="$SIMS/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test"

export HB_PATH_PP="$SIMS/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test"

# Separate output directory to preserve existing polar plots
export HB_OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/Polar_3D_minus_PP_morning_afternoon

# MSWX timestamps represent the beginning of 3-hour intervals.
# Shift by 1.5 hours for interval-midpoint daypart classification.
# This does NOT change model radiation or solar geometry.
export HB_TIME_SHIFT_HOURS=1.5

# Memory and parallel processing controls
export HB_TIME_CHUNK=128
export HB_ALLOW_PARTIAL=0

cd "$REPO"

mkdir -p logs
mkdir -p "$HB_OUTDIR"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found: $SCRIPT"
    exit 1
fi

echo "Python:  $CONDA_PREFIX/bin/python"
echo "Script:  $SCRIPT"
echo "3D run:  $HB_PATH_3D"
echo "PP run:  $HB_PATH_PP"
echo "Output:  $HB_OUTDIR"
echo "Workers: $SLURM_CPUS_PER_TASK"
echo "Time shift: $HB_TIME_SHIFT_HOURS hours"

"$CONDA_PREFIX/bin/python" "$SCRIPT"

echo "Done: $(date)"
