#!/bin/bash
#SBATCH --job-name=HB_3d_MODSCAG
#SBATCH --partition=amem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --qos=mem-normal
#SBATCH --output=logs/3d_modscag_fsno-%j.out
#SBATCH --error=logs/3d_modscag_fsno-%j.err

set -eo pipefail

echo "Start: $(date)"
echo "Node:  $SLURM_NODELIST"
echo "Job:   $SLURM_JOB_ID"

module purge
module load slurm/alpine
module load anaconda

set -u

export CONDA_PREFIX=/projects/battobrah@xsede.org/software/anaconda/envs/HB3
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PROJ_LIB="$CONDA_PREFIX/share/proj"
export PROJ_DATA="$CONDA_PREFIX/share/proj"
export GDAL_DATA="$CONDA_PREFIX/share/gdal"

REPO=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev

SCRIPT="$REPO/full_domain_plot_scripts/plot_postprocessed_fsno_vs_modscaq.py"

EDIR=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint

MODSCAG_DIR=/scratch/alpine/battobrah@xsede.org/HB_NMP_full_domain_comparison_data/modscag_domain_extract_old

OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS_midpoint/3d_vs_MODSCAG

cd "$REPO"

mkdir -p "$OUTDIR"
mkdir -p logs

echo "Python: $CONDA_PREFIX/bin/python"
echo "Python version: $("$CONDA_PREFIX/bin/python" --version)"
echo "Script: $SCRIPT"
echo "3d experiment: $EDIR"
echo "MODSCAG: $MODSCAG_DIR"
echo "Output: $OUTDIR"

"$CONDA_PREFIX/bin/python" "$SCRIPT" \
  --edir "$EDIR" \
  --modscag-dir "$MODSCAG_DIR" \
  --start-year 2014 \
  --end-year 2024 \
  --model-variable fsno \
  --modscag-variable snow_fraction \
  --bins 50 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --dpi 400 \
  --outdir "$OUTDIR"

echo "Done: $(date)"