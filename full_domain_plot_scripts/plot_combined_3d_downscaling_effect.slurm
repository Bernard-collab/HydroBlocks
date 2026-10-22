#!/bin/bash

#SBATCH --job-name=3D_downscaling_pairs
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/3d_downscaling_pairs-%j.out
#SBATCH --error=logs/3d_downscaling_pairs-%j.err

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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev
SCRIPT="$REPO/full_domain_plot_scripts/plot_combined_3d_downscaling_effect.py"

EDIR_3D=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test

EDIR_BASELINE=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_no_downscaled_test

OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/3D_downscaled_minus_PP_no_downscaling

cd "$REPO"
mkdir -p logs "$OUTDIR"

echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job: $SLURM_JOB_ID"
echo "Python: $CONDA_ENV/bin/python"
echo "Workers: $SLURM_CPUS_PER_TASK"

"$CONDA_ENV/bin/python" "$SCRIPT" \
    --edir-3d "$EDIR_3D" \
    --edir-pp "$EDIR_BASELINE" \
    --start-year 2014 \
    --end-year 2024 \
    --pairs shortwave longwave snow_state \
    --workers "$SLURM_CPUS_PER_TASK" \
    --soil-layer 0 \
    --dpi 400 \
    --outdir "$OUTDIR"

echo "Done: $(date)"