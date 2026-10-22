#!/bin/bash
#SBATCH --job-name=HB_3d_SNODAS
#SBATCH --partition=amem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --qos=mem-normal
#SBATCH --output=logs/3d_snodas_swe-%j.out
#SBATCH --error=logs/3d_snodas_swe-%j.err

set -eo pipefail

echo "Start: $(date)"
echo "Node:  $SLURM_NODELIST"
echo "Job:   $SLURM_JOB_ID"

module purge
module load slurm/alpine
module load anaconda

set -u

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
SCRIPT="$REPO/full_domain_plot_scripts/plot_postprocessed_swe_vs_snodas.py"

EDIR=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test_forcing_midpoint

SNODAS_FILE=/scratch/alpine/battobrah@xsede.org/HB_NMP_full_domain_comparison_data/snodas_swe_seasonal_30arcsec.nc

# Enter the verified source CRS-confirm crs with enrico.
SNODAS_CRS="EPSG:4326"

OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS_midpoint/3d_vs_SNODAS

START_YEAR=2014
END_YEAR=2016

cd "$REPO"
mkdir -p "$OUTDIR" logs

if [[ -z "$SNODAS_CRS" ]]; then
    echo "Set SNODAS_CRS to the verified source CRS before running."
    exit 1
fi

echo "Python: $CONDA_ENV/bin/python"
echo "Script: $SCRIPT"
echo "3D experiment: $EDIR"
echo "SNODAS: $SNODAS_FILE"
echo "Source CRS: $SNODAS_CRS"
echo "Years: $START_YEAR–$END_YEAR"
echo "Output: $OUTDIR"

"$CONDA_ENV/bin/python" "$SCRIPT" \
  --edir "$EDIR" \
  --snodas-file "$SNODAS_FILE" \
  --snodas-crs "$SNODAS_CRS" \
  --start-year "$START_YEAR" \
  --end-year "$END_YEAR" \
  --diff-limit 200 \
  --bins 50 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --dpi 400 \
  --outdir "$OUTDIR"

echo "Done: $(date)"