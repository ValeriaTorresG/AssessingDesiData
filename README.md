# Assessing the quality of DESI Spectroscopic Survey

Key features include:

- [x]  Load the combined dataset from the HDF5 file.
- [x]  Perform any necessary preprocessing (e.g. padding if a brz grid is used).
- [x]  Apply a U-MAP algorithm to reduce the high-dimensional spectral data to 2D for visualization. 
- [x]  Identify clusters or groupings of spectra using FoF.
- [x]  Plot the results of the UMAP projection, each point represents an object’s spectrum in the low-dimensional space. Points are color-coded by object category (e.g., gal, qso, std).
## Outliers identified

 - Some outliers found are displayed [here](https://valeriatorresg.github.io/AssessingDesiData/).


## Running Tests

### On NERSC

SLURM to submit jobs: below is an example batch script (see ```/jobs/nersc_array.sh```):

```bash
#!/bin/bash
#SBATCH --job-name=qa_umap_outliers
#SBATCH --account=desi
#SBATCH --partition=cron
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:04:00
#SBATCH --mem=4G

module load python/3.12

BASE_DIR=/global/cfs/cdirs/desi/spectro/redux/jura/tiles/cumulative
LOGDIR=/pscratch/sd/v/vtorresg/umap_analysis/data/logs
mkdir -p "$LOGDIR"

TILE=$(ls "$BASE_DIR" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
NIGHT=$(ls "$BASE_DIR/$TILE" | head -n1)

OUTFILE=${LOGDIR}/${TILE}.out
ERRFILE=${LOGDIR}/${TILE}.err

srun python /global/homes/v/vtorresg/AssessingDesiData/src/scripts/run_pipeline.py \
     --tile    "${TILE}" \
     --night   "${NIGHT}" \
     >"$OUTFILE" 2>"$ERRFILE"
```

### Local
Uses local DESI Data Release 1 (DR1) files under ```/data/desi_data/{night}``` 

```bash
./jobs/run.sh \
  --tile {tile_id} \
  --night {night} \
  --base-dir /data/desi_data \
  --processed-dir /data/processed \
  --band brz \
  --n_neighbors 45 \
  --min_dist 1.0 \
  --n_components 2 \
  --link-length 0.45 \
  --min-cluster-size 5 \
  --workers 8
```

- ```--tile``` : DESI tile ID (e.g. 10256)

- ```--night``` : Observation date (e.g. 20211110)

- ```--base-dir``` : Root folder containing DR1 data (/data/desi_data)

- ```--processed-dir``` : Folder where processed HDF5 and plots are saved (/data/processed)

- ```--band``` : Bands to process (b, r, z, or combined brz)

- ```--n_neighbors``` : Number of neighbors for UMAP (default: 45)

- ```--min_dist``` : UMAP minimum distance parameter (default: 1.0)

- ```--n_components``` : Dimensionality of UMAP embedding (2 or 3)

- ```--link-length``` : Radius for Friends-of-Friends clustering (default: 0.45)

- ```--min-cluster-size``` : Minimum cluster size before flagging as outlier (default: 5)

- ```--workers``` : Number of parallel workers (e.g. CPU cores)



## Procedure

- **Directory traversal**: Recursively scan `data/desi_data` to locate all `coadd-<tile>-<night>.fits` files organized by tile and night.  

- **File reading**: For each FITS file, use `desispec.io.read_spectra` (or `astropy.io.fits`) to load wavelength and flux arrays for bands **b**, **r**, and **z**.  

- **Data filtering**: Apply fiber-status and target-flag masks to remove bad fibers or non-science targets before further processing.  

- **Flux & wavelength extraction**: Extract and store per-object flux arrays (`flux[b/r/z]`) alongside their corresponding wavelength grids (`wave[b/r/z]`).  

- **Padding alignment**: Zero-pad or truncate each band’s flux/wavelength arrays so that all spectra share a common length, enabling matrix stacking.  

- **Matrix construction**: Stack padded flux arrays into a 2D matrix of shape `(n_objects, n_wavelengths_total)` and similarly assemble metadata arrays (tile IDs, object IDs, target types).  

- **HDF5 export**: Write the combined flux matrix, wavelength grid, and metadata into a single HDF5 file (`desi_spectra.h5`) with named datasets for efficient I/O.  

- **Normalization (not used!)**: Load the HDF5 file, optionally normalize each spectrum (e.g., by its median flux) to mitigate brightness differences before dimensionality reduction.  

- **UMAP embedding**: Use `umap.UMAP` to project the high-dimensional flux matrix into 2D or 3D, capturing spectral similarity in a low-dimensional space.  

- **FoF clustering & outlier detection**: Build a radius-neighbors graph on the UMAP embedding, apply `scipy.sparse.csgraph.connected_components` (Friends-of-Friends) to label clusters, and flag small clusters or singletons as outliers.  

- **Visualization**: Generate and save plots—UMAP scatter colored by object type, example spectra overlays, and tile-specific summary figures—into the `plots/` directory for inspection.  
