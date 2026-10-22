#!/bin/bash
#SBATCH --job-name=HB_static_plot
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=cpu-normal
#SBATCH --output=logs/static_plot-%j.out
#SBATCH --error=logs/static_plot-%j.err

set -e

echo "Start: $(date)"
echo "Node:  $SLURM_NODELIST"

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
SCRIPT="$REPO/full_domain_plot_scripts/plot_hb_static_domain.py"

cd "$REPO"

echo "Python: $CONDA_PREFIX/bin/python"
echo "Python version: $("$CONDA_PREFIX/bin/python" --version)"
echo "Script: $SCRIPT"

"$CONDA_PREFIX/bin/python" "$SCRIPT" \
  --edir "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test" \
  --out "/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/new_3d_full_domain_res_terrain_plot_11yr_100hru.png" \
  --cols 2 \
  --buffer 0 \
  --border 0.02 \
  --dpi 400

echo "Done: $(date)"