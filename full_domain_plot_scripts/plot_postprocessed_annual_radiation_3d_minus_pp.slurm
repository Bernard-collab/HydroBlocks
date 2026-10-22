#!/bin/bash

#SBATCH --job-name=HB_annual_radiation
#SBATCH --partition=amem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --qos=mem-normal
#SBATCH --output=logs/annual_radiation-%j.out
#SBATCH --error=logs/annual_radiation-%j.err

set -euo pipefail

echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job: $SLURM_JOB_ID"

module purge
module load slurm/alpine
module load anaconda

export CONDA_PREFIX=/projects/battobrah@xsede.org/software/anaconda/envs/HB3
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PROJ_LIB="$CONDA_PREFIX/share/proj"
export PROJ_DATA="$CONDA_PREFIX/share/proj"
export GDAL_DATA="$CONDA_PREFIX/share/gdal"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev
SCRIPT="$REPO/full_domain_plot_scripts/plot_postprocessed_annual_radiation_3d_minus_pp.py"

EDIR_3D=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint
EDIR_PP=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint
OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS__midpoint/3D_minus_PP/Annual_Radiation

START_YEAR=2014
END_YEAR=2016

cd "$REPO"
mkdir -p "$OUTDIR"

"$CONDA_PREFIX/bin/python" -u "$SCRIPT" \
    --edir-3d "$EDIR_3D" \
    --edir-pp "$EDIR_PP" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --dpi 400 \
    --outdir "$OUTDIR"

echo "Done: $(date)"