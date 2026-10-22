#!/bin/bash
#SBATCH --job-name=plot_study_domain
#SBATCH --output=plot_study_domain_%j.out
#SBATCH --error=plot_study_domain_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000m
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --account=rmacc-general


# ---- environment ----
module purge
module load anaconda
conda activate geo_preproc

# ---- paths (edit these) ----
SCRIPT=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev/plot_study_domain.py
DEM=/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test/postprocess/dem.vrt
OUT=/home/battobrah@xsede.org/HydroBlocks_Enrico_dev/study_domain_map.png

# ---- run ----
python "$SCRIPT" --dem "$DEM" --out "$OUT" --dpi 300

echo "Done. Output written to: $OUT"
