#!/bin/bash
#SBATCH --job-name=qa_umap_outliers2
#SBATCH --account=desi
#SBATCH --partition=cron
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:04:00
#SBATCH --mem=4G
#SBATCH -C cron
#SBATCH --array=1-37%15

module load python/3.12

BASE_DIR=/global/cfs/cdirs/desi/spectro/redux/jura/tiles/cumulative
LOGDIR=/pscratch/sd/v/vtorresg/umap_analysis/data/logs
FAILED_LIST=/global/homes/v/vtorresg/AssessingDesiData/jobs/failed_jobs.txt

RAW=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FAILED_LIST")
TILE=${RAW%.err}
NIGHT=$(ls "${BASE_DIR}/${TILE}" | head -n1)

OUTFILE=${LOGDIR}/${TILE}.out
ERRFILE=${LOGDIR}/${TILE}.err

srun python /global/homes/v/vtorresg/AssessingDesiData/src/scripts/run_pipeline.py \
     --tile    "${TILE}" \
     --night   "${NIGHT}" \
     >"$OUTFILE" 2>"$ERRFILE"