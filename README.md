# Assessing the quality of DESI Spectroscopic Survey


## Outliers identified

 - Some outliers found are displayed [here](https://valeriatorresg.github.io/AssessingDesiData/).


## Running Tests

### On NERSC

SLURM to submit jobs: below is an example batch script (see `/jobs/run_jobs.sbatch`):

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
Uses local DESI Data Release 1 (DR1) files under `/data/desi_data/{night}` 

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
  --min-cluster-size 5
```

- `--tile` : DESI tile ID (e.g. 10256)

- `--night` : Observation date (e.g. 20211110)

- `--base-dir` : Root folder containing data (/data/desi_data)

- `--processed-dir` : Folder where processed HDF5 and plots are saved (/data/processed)

- `--band` : Bands to process (b, r, z, or combined brz)

- `--n_neighbors` : Number of neighbors for UMAP (default: 45)

- `--min_dist` : UMAP minimum distance parameter (default: 1.0)

- `--n_components` : Dimensionality of UMAP embedding (2 or 3)

- `--link-length` : Radius for Friends-of-Friends clustering (default: 0.45)

- `--min-cluster-size` : Minimum cluster size before flagging as outlier (default: 5)


## Summary

- **Data loading & filtering**  
  Recursively scan for `coadd-<tile>-<night>.fits`, load wavelength and flux arrays, and apply masks to remove bad fibers/targets.

- **Matrix preparation**  
  Extract per-band flux & wavelength arrays, pad to a common length, then stack into a 2D flux matrix with associated metadata.

- **Dimensionality reduction & clustering**  
  Use UMAP to embed the combined flux matrix into 2D/3D, build a radius-neighbors graph, apply FoF clustering, and flag small clusters as outliers.

- **Export & visualization**  
  Write the flux matrix, wavelength grid, and metadata to a single HDF5 file, and save UMAP scatter plots, spectral overlays, and tile-specific summaries in `plots/`.
