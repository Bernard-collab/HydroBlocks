#!/bin/bash
#SBATCH --job-name=static_maps
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --qos=normal
#SBATCH --output=logs/static_maps-%j.out
#SBATCH --error=logs/static_maps-%j.err

set -e

module purge
module load slurm/alpine

export CONDA_PREFIX=/projects/battobrah@xsede.org/software/anaconda/envs/HB3
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PROJ_LIB=$CONDA_PREFIX/share/proj
export PROJ_DATA=$CONDA_PREFIX/share/proj

cd /home/battobrah@xsede.org/HydroBlocks_Enrico_dev
mkdir -p logs

$CONDA_PREFIX/bin/python static_maps_100hru.py
