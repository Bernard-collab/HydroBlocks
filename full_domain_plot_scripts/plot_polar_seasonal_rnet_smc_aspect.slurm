#!/bin/bash

#SBATCH --job-name=HB_polar_seasonal
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/polar_seasonal-%j.out
#SBATCH --error=logs/polar_seasonal-%j.err

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

# One thread per CID worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

export HB_TIME_CHUNK=128
export HB_ALLOW_PARTIAL=0

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev
SCRIPT="$REPO/full_domain_plot_scripts/plot_polar_seasonal_rnet_smc_aspect.py"

SIMS=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations

export HB_PATH_3D="$SIMS/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint"
export HB_PATH_PP="$SIMS/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint"

export HB_OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS_midpoint/Polar_seasonal_Rnet_SMC_aspect

export HB_START_YEAR=2014
export HB_END_YEAR=2016

cd "$REPO"
mkdir -p "$HB_OUTDIR"

# Check paths and required Python packages
test -f "$SCRIPT"
test -d "$HB_PATH_3D"
test -d "$HB_PATH_PP"

"$CONDA_PREFIX/bin/python" -c \
    "import numpy, netCDF4, rasterio, matplotlib, scipy"

echo "Python:  $CONDA_PREFIX/bin/python"
echo "Script:  $SCRIPT"
echo "3D run:  $HB_PATH_3D"
echo "PP run:  $HB_PATH_PP"
echo "Output:  $HB_OUTDIR"
echo "Workers: $SLURM_CPUS_PER_TASK"

"$CONDA_PREFIX/bin/python" -u "$SCRIPT"

echo "Done: $(date)"