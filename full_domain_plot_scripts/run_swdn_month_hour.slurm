#!/bin/bash

#SBATCH --job-name=SWDN_month_hour
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/swdn_month_hour-%j.out
#SBATCH --error=logs/swdn_month_hour-%j.err

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
SCRIPT="$REPO/full_domain_plot_scripts/plot_cid_swdn_month_hour.py"

EDIR_3D=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test
EDIR_PP=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test
OUTDIR=/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/3D_minus_PP/SWDN_month_hour

# Raw output file inside each CID directory; override using sbatch --export.
RAW_FILENAME=${RAW_FILENAME:-2016-01-01.nc}
# Raw timestamps are UTC; use Mountain Standard Time throughout.
UTC_OFFSET_HOURS=${UTC_OFFSET_HOURS:--7}

cd "$REPO"
mkdir -p "$OUTDIR"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Missing Python script: $SCRIPT" >&2
    exit 1
fi

echo "Start: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job: $SLURM_JOB_ID"
echo "Python: $CONDA_ENV/bin/python"
echo "Raw file: $RAW_FILENAME; 1600 CIDs; time offset: $UTC_OFFSET_HOURS hours"

"$CONDA_ENV/bin/python" "$SCRIPT" \
    --edir-3d "$EDIR_3D" \
    --edir-pp "$EDIR_PP" \
    --filename "$RAW_FILENAME" \
    --months 12 7 12 7 12 7 12 7 \
    --hours 8 8 11 11 14 14 17 17 \
    --utc-offset-hours "$UTC_OFFSET_HOURS" \
    --dpi 400 \
    --outdir "$OUTDIR"

echo "Done: $(date)"
