HydroBlocks
==========

The following steps walk through how to install HydroBlocks

**1. Clone the HydroBlocks repository.**

```
git clone https://github.com/chaneyn/HydroBlocks.git
cd HydroBlocks
```

**2. Create a conda environment named HB from the spec-file. Note that the only current spec-file in the repository is for a linux64 machine.** 

```
conda create --name HB --file yml/nots_environment.txt
source activate HB
pip install git+https://github.com/chaneyn/geospatialtools@dev_nate_2023
pip install psutil
```
Alternatively install the packages directly from anaconda:

```
conda create --name HB
source activate HB
conda install -c conda-forge netcdf4 gdal geos jpeg scikit-learn numpy scipy h5py matplotlib cartopy mpi4py zarr opencv pandas numba xarray rioxarray fiona rasterio time
pip install git+https://github.com/chaneyn/geospatialtools@dev_nate_2023
pip install psutil
```

**3. Install HydroBlocks.**

```
python setup.py 
```

**4. Using the code in Alpine.**

```
module load anaconda
```

```
conda activate HB5
```


```
./HB -m script3d.json -t cluster
```

```
./HB -m script3d.json -t model
```

For postprocessing, use this env where i fixed matplotlib - broken for some reason

```
conda activate HBPP2
```