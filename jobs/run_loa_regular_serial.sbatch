#!/bin/bash
#SBATCH --job-name=qa_loa_regular_serial
#SBATCH --account=desi
#SBATCH --partition=regular_milan_ss11
#SBATCH --constraint=cpu
#SBATCH --output=/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/logs/loa_regular_%j.out
#SBATCH --error=/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/logs/loa_regular_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -uo pipefail

source ~/venvs/my-umap-env/bin/activate

BASE_DIR=/global/cfs/cdirs/desi/spectro/redux/loa/tiles/cumulative
OUTROOT=/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data
FAILED_LIST=${FAILED_LIST:-/global/homes/v/vtorresg/AssessingDesiData/jobs/loa_failed_tiles.txt}
START_LINE=${START_LINE:-1}
END_LINE=${END_LINE:-0}

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

if [[ ! -f "${FAILED_LIST}" ]]; then
    echo "Missing failed tile list: ${FAILED_LIST}" >&2
    exit 2
fi

line_no=0
processed=0
failures=0
skipped=0

while IFS= read -r TILE || [[ -n "${TILE}" ]]; do
    TILE=${TILE%%#*}
    TILE=${TILE//[[:space:]]/}
    [[ -z "${TILE}" ]] && continue

    line_no=$((line_no + 1))
    if (( line_no < START_LINE )); then
        skipped=$((skipped + 1))
        continue
    fi
    if (( END_LINE > 0 && line_no > END_LINE )); then
        break
    fi

    if [[ ! -d "${BASE_DIR}/${TILE}" ]]; then
        echo "$(date -Is) missing tile directory: ${BASE_DIR}/${TILE}" >&2
        failures=$((failures + 1))
        continue
    fi

    NIGHT=$(ls "${BASE_DIR}/${TILE}" | sort | tail -n1)
    if [[ -z "${NIGHT}" ]]; then
        echo "$(date -Is) no night found for tile ${TILE}" >&2
        failures=$((failures + 1))
        continue
    fi

    URL_DIR=${OUTROOT}/inspector_urls/${TILE}
    mkdir -p "${URL_DIR}"

    OUTFILE=${LOGDIR}/${TILE}.out
    ERRFILE=${LOGDIR}/${TILE}.err

    echo "$(date -Is) start line=${line_no} tile=${TILE} night=${NIGHT}"
    if srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        python /global/homes/v/vtorresg/AssessingDesiData/src/scripts/run_pipeline.py \
            --tile          "${TILE}" \
            --night         "${NIGHT}" \
            --base-dir      "${BASE_DIR}" \
            --processed-dir "${PROCESSED_DIR}" \
            --band          brz \
            --fiber_plot    "${PLOT_DIR}" \
            --out_txt       "${TXT_DIR}" \
            --out_log       "${URL_DIR}" \
            >"${OUTFILE}" 2>"${ERRFILE}"; then
        processed=$((processed + 1))
        echo "$(date -Is) done tile=${TILE}"
    else
        rc=$?
        failures=$((failures + 1))
        echo "$(date -Is) failed tile=${TILE} rc=${rc} err=${ERRFILE}" >&2
    fi
done < "${FAILED_LIST}"

echo "$(date -Is) summary processed=${processed} failures=${failures} skipped=${skipped}"

if (( failures > 0 )); then
    exit 1
fi