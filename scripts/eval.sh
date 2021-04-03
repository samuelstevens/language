#!/usr/bin/env bash

set -euo pipefail

DATA_DIR=${DATA_DIR:-"data"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"xsp_experiment_run"}
OUTPUT_VOCAB_FILE=${OUTPUT_VOCAB_FILE:-"${EXPERIMENT_DIR}/assets/output_vocab.txt"}
PREDICTIONS_DIR="${EXPERIMENT_DIR}/trial${TRIAL_NUM}_predictions"

# First argument is the dataset name, second is the split name
# run_inference spider dev
# run_inference yelp dev
function evaluate {
    python -m language.xsp.evaluation.official_evaluation \
      --predictions_filepath=${PREDICTIONS_DIR}/${1}_${2}_instructions.json \
      --output_filepath=${PREDICTIONS_DIR}/${1}_${2}_predictions.txt \
      --cache_filepath=${PREDICTIONS_DIR}/${1}_${2}_cache.json \
      --update_cache=True \
      --format=json | tail -n 1 | \
      jq ". += { \"name\": \"${1}\" }" 
}

output_json_filepath=${PREDICTIONS_DIR}/predictions_summary.jsonl

evaluate spider dev > ${output_json_filepath}

evaluate atis dev >> ${output_json_filepath}
evaluate academic dev >> ${output_json_filepath}
# evaluate advising dev
evaluate geoquery dev >> ${output_json_filepath}
evaluate imdb dev >> ${output_json_filepath}
evaluate restaurants dev >> ${output_json_filepath}
evaluate scholar dev >> ${output_json_filepath}
evaluate yelp dev >> ${output_json_filepath}
