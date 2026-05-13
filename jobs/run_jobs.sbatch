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

TILE=$(ls "$BASE_DIR" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
NIGHT=$(ls "$BASE_DIR/$TILE" | sort | tail -n1)

OUTFILE=${LOGDIR}/${TILE}.out
ERRFILE=${LOGDIR}/${TILE}.err

srun python /global/homes/v/vtorresg/AssessingDesiData/src/scripts/run_pipeline.py \
     --tile    "${TILE}" \
     --night   "${NIGHT}" \
     >"$OUTFILE" 2>"$ERRFILE"