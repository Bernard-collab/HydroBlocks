#!/bin/bash

#SBATCH --job-name=HB_3D_PP_pairs
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/3d_pp_pairs-%j.out
#SBATCH --error=logs/3d_pp_pairs-%j.err

set -euo pipefail

module purge
module load slurm/alpine
module load anaconda

CONDA_ENV=/projects/battobrah@xsede.org/software/anaconda/envs/HB3

export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PROJ_LIB="$CONDA_ENV/share/proj"
export PROJ_DATA="$CONDA_ENV/share/proj"
export GDAL_DATA="$CONDA_ENV/share/gdal"

# Prevent each worker from starting additional numerical-library threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev
SCRIPT="$REPO/full_domain_plot_scripts/plot_seasonal_3d_minus_pp_parallel.py"

EDIR_3D=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint

EDIR_PP=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint

OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS_midpoint/3D_minus_PP

cd "$REPO"
mkdir -p logs "$OUTDIR"

echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job: $SLURM_JOB_ID"
echo "Python: $CONDA_ENV/bin/python"
echo "Workers: $SLURM_CPUS_PER_TASK"

"$CONDA_ENV/bin/python" "$SCRIPT" \
    --edir-3d "$EDIR_3D" \
    --edir-pp "$EDIR_PP" \
    --start-year 2014 \
    --end-year 2016 \
    --pairs shortwave longwave soil_moisture snow_amount snow_state \
            near_surface_temperature surface_temperature turbulent_fluxes surface_water \
    --workers "$SLURM_CPUS_PER_TASK" \
    --soil-layer 0 \
    --dpi 400 \
    --outdir "$OUTDIR"

echo "Done: $(date)"