#!/bin/bash
#SBATCH --job-name=qa_loa
#SBATCH --account=desi
#SBATCH --partition=cron
#SBATCH --output=/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/logs/loa_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/logs/loa_%A_%a.err
#SBATCH --time=00:04:00
#SBATCH --mem=4G
#SBATCH -C cron
#SBATCH --array=10001-14199%200

set -euo pipefail

source ~/venvs/my-umap-env/bin/activate

BASE_DIR=/global/cfs/cdirs/desi/spectro/redux/loa/tiles/cumulative
OUTROOT=/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data
LOGDIR=${OUTROOT}/logs
PROCESSED_DIR=${OUTROOT}/processed
PLOT_DIR=${OUTROOT}/plots
TXT_DIR=${OUTROOT}/text_files
NUMBA_CACHE_DIR=${OUTROOT}/numba_cache
MPLCONFIGDIR=${OUTROOT}/mpl_config
XDG_CACHE_HOME=${OUTROOT}/xdg_cache

export NUMBA_CACHE_DIR MPLCONFIGDIR XDG_CACHE_HOME

mkdir -p "${LOGDIR}" "${PROCESSED_DIR}" "${PLOT_DIR}" "${TXT_DIR}" \
         "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

TILE=$(ls "${BASE_DIR}" | sort -n | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [[ -z "${TILE}" ]]; then
    echo "No tile found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

NIGHT=$(ls "${BASE_DIR}/${TILE}" | sort | tail -n1)
if [[ -z "${NIGHT}" ]]; then
    echo "No night found for tile ${TILE}" >&2
    exit 1
fi

URL_DIR=${OUTROOT}/inspector_urls/${TILE}
mkdir -p "${URL_DIR}"

OUTFILE=${LOGDIR}/${TILE}.out
ERRFILE=${LOGDIR}/${TILE}.err

srun python /global/homes/v/vtorresg/AssessingDesiData/src/scripts/run_pipeline.py \
     --tile          "${TILE}" \
     --night         "${NIGHT}" \
     --base-dir      "${BASE_DIR}" \
     --processed-dir "${PROCESSED_DIR}" \
     --band          brz \
     --fiber_plot    "${PLOT_DIR}" \
     --out_txt       "${TXT_DIR}" \
     --out_log       "${URL_DIR}" \
     >"${OUTFILE}" 2>"${ERRFILE}"