#!/bin/bash

#SBATCH --job-name=HB_SMAP_SMC
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/smc_smap_am_pm-%j.out
#SBATCH --error=logs/smc_smap_am_pm-%j.err

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

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev

SCRIPT="$REPO/full_domain_plot_scripts/plot_postprocessed_smc_vs_smap_am.py"

EDIR=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test

SMAP_DIR="/scratch/alpine/battobrah@xsede.org/HB_NMP_full_domain_comparison_data/smap_domain_extract"

OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/SMAP_comparison

cd "$REPO"

mkdir -p "$OUTDIR"
mkdir -p logs

echo "Python: $CONDA_PREFIX/bin/python"
echo "Python version: $("$CONDA_PREFIX/bin/python" --version)"
echo "Script: $SCRIPT"
echo "Experiment: $EDIR"
echo "SMAP: $SMAP_DIR"
echo "Output: $OUTDIR"

"$CONDA_PREFIX/bin/python" "$SCRIPT" \
  --edir "$EDIR" \
  --smap-dir "$SMAP_DIR" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --start-year 2016 \
  --end-year 2023 \
  --dpi 400 \
  --outdir "$OUTDIR"

echo "Done: $(date)"