#!/bin/bash

#SBATCH --job-name=HB_terrain_scatter
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/terrain-scatter-%j.out
#SBATCH --error=logs/terrain-scatter-%j.err

set -euo pipefail

module purge
module load slurm/alpine
module load anaconda

CONDA_ENV=/projects/battobrah@xsede.org/software/anaconda/envs/HB3
REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev
SCRIPT="$REPO/plot_scatter_3d_minus_pp_vs_vars_parallel.py"

EDIR_3D=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test

EDIR_PP=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test

OUTDIR=/scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/validation_plots/terrain_scatter

export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO"
mkdir -p logs "$OUTDIR"

echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Workers: $SLURM_CPUS_PER_TASK"

"$CONDA_ENV/bin/python" "$SCRIPT" \
    --edir-3d "$EDIR_3D" \
    --edir-pp "$EDIR_PP" \
    --variables trad \
    --start-year 2014 \
    --end-year 2024 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --dpi 400 \
    --outdir "$OUTDIR"

echo "Done: $(date)"