#!/bin/bash
#SBATCH --job-name=qa_umap_outliers
#SBATCH --account=desi
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:04:00
#SBATCH --mem=32G
#SBATCH --array=1-1%1
#SBATCH -C cpu

module load python/3.12

BASE_DIR=/global/cfs/cdirs/desi/spectro/redux/jura/tiles/cumulative
LOGDIR=/pscratch/sd/v/vtorresg/umap_analysis/data/logs
mkdir -p "$LOGDIR"

TILE=$(ls "$BASE_DIR" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
NIGHT=$(ls "$BASE_DIR/$TILE" | head -n1)

OUTFILE=${LOGDIR}/${TILE}_${NIGHT}.out
ERRFILE=${LOGDIR}/${TILE}_${NIGHT}.err

srun python /global/homes/v/vtorresg/AssessingDesiData/src/scripts/run_pipeline.py \
     --tile    "${TILE}" \
     --night   "${NIGHT}" \
     >"$OUTFILE" 2>"$ERRFILE"