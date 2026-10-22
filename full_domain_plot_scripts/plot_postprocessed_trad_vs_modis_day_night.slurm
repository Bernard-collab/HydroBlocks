#!/bin/bash

#SBATCH --job-name=HB_MODIS_LST
#SBATCH --partition=amem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --qos=mem-normal
#SBATCH --output=logs/trad_modis_day_night-%j.out
#SBATCH --error=logs/trad_modis_day_night-%j.err

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
SCRIPT="$REPO/full_domain_plot_scripts/plot_postprocessed_trad_vs_modis_day_night.py"

EDIR=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint

MODIS_DIR=/projects/battobrah@xsede.org/HB_NOAH-MP_Backup_DATA/MODIS_LST_2014_2024

OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS_midpoint/Surface_Temp_Plots

cd "$REPO"
mkdir -p "$OUTDIR"

echo "Python: $CONDA_PREFIX/bin/python"
echo "Python version: $("$CONDA_PREFIX/bin/python" --version)"
echo "Script: $SCRIPT"
echo "Experiment: $EDIR"
echo "MODIS: $MODIS_DIR"
echo "Output: $OUTDIR"

"$CONDA_PREFIX/bin/python" "$SCRIPT" \
  --edir "$EDIR" \
  --modis-dir "$MODIS_DIR" \
  --start-year 2014 \
  --end-year 2024 \
  --modis-units celsius \
  --assume-model-step-hours 3 \
  --day-local-hour 10.5 \
  --night-local-hour 22.5 \
  --minimum-day-coverage 0.9 \
  --dpi 400 \
  --outdir "$OUTDIR"

echo "Done: $(date)"