#!/usr/bin/env bash
#
# Alignment and feature extraction for the BIND face cryptosystem.
#
#   ./run_standalone.sh              # extract features for every dataset
#   ./run_standalone.sh align        # align one dataset first
#   ./run_standalone.sh extract      # extraction only (default)
#
# Override any of the settings below from the environment, e.g.
#   GPU=2 MODEL=--use-adaface-cvl ./run_standalone.sh
#   DATASETS="lfw-a cfp-frontal" ./run_standalone.sh

set -euo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────

PROJECT_ROOT="/egr/research-sprintai/benja161/AgenticAI/agentic_idOBO"
IMAGESETS="${PROJECT_ROOT}/datasets/imagesets"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/datasets/data_extracts/face_features}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Recognition backbone: --use-adaface-cvl | --use-arcface-cvl | --use-vitkprpe-cvl
MODEL="${MODEL:---use-vitkprpe-cvl}"

# Datasets to extract. Keys must exist in DATASETS in facefeature_extractor.py.
DATASETS="${DATASETS:-lfw-a cfp-frontal multipie celeba-hq-frontal casia-face}"

# Maps a dataset key to its image root, so --dataset-root can be passed
# explicitly rather than relying on the table compiled into the extractor.
dataset_root() {
    case "$1" in
        cfp-frontal)       echo "${IMAGESETS}/CFP-frontal/"        ;;
        lfw-a)             echo "${IMAGESETS}/LFW-a-set/"          ;;
        multipie)          echo "${IMAGESETS}/Multi_PIE/"          ;;
        celeba-hq)         echo "${IMAGESETS}/CelebA-HQ-set/"      ;;
        celeba-hq-frontal) echo "${IMAGESETS}/CelebA-HQ-frontal/"  ;;
        casia-face)        echo "${IMAGESETS}/CASIA-webface/"      ;;
        *) echo "unknown dataset key: $1" >&2; return 1            ;;
    esac
}

cd "$(dirname "$0")"

# ── Alignment ────────────────────────────────────────────────────────────────
# Crops faces to 112x112 into <dataset-root>/aligned/, mirroring the input tree.
# Run once per dataset before extracting; skip if aligned/ is already populated.

run_align() {
    local key="${ALIGN_DATASET:-celeba-hq-frontal}"
    local root; root="$(dataset_root "$key")"

    if [[ ! -d "${root}/images" ]]; then
        echo "!! no images/ under ${root} — nothing to align" >&2
        return 1
    fi

    echo "== aligning ${key}"
    CUDA_VISIBLE_DEVICES="${GPU}" python cvlface_align_faces.py \
        --data-root "${root}/images/" \
        --save-root "${root}/aligned/" \
        --aligner-id minchul/cvlface_DFA_mobilenet
}

# ── Feature extraction ───────────────────────────────────────────────────────
# Writes <output-root>/<dataset>/<model>_features.pt + image_filenames.txt.

run_extract() {
    mkdir -p "${OUTPUT_ROOT}"

    local failed=()
    for key in ${DATASETS}; do
        local root
        if ! root="$(dataset_root "${key}")"; then
            failed+=("${key}")
            continue
        fi

        if [[ ! -d "${root}/aligned" ]]; then
            echo "!! skipping ${key}: no aligned/ under ${root}" >&2
            failed+=("${key}")
            continue
        fi

        echo "== extracting ${key}  (${MODEL})"
        # Keep going if one dataset fails, so a long sweep is not lost.
        if ! CUDA_VISIBLE_DEVICES="${GPU}" python facefeature_extractor.py \
                --dataset "${key}" \
                --dataset-root "${root}" \
                --output-root "${OUTPUT_ROOT}" \
                --batch-size "${BATCH_SIZE}" \
                ${MODEL} \
                --cuda --verbose; then
            echo "!! ${key} failed" >&2
            failed+=("${key}")
        fi
    done

    if (( ${#failed[@]} )); then
        echo
        echo "finished with failures: ${failed[*]}" >&2
        return 1
    fi
    echo
    echo "all datasets extracted to ${OUTPUT_ROOT}"
}

# ── Entry point ──────────────────────────────────────────────────────────────

case "${1:-extract}" in
    align)   run_align   ;;
    extract) run_extract ;;
    all)     run_align && run_extract ;;
    *) echo "usage: $0 [align|extract|all]" >&2; exit 2 ;;
esac
