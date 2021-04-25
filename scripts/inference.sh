#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} 

DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_VOCAB_FILE=${OUTPUT_VOCAB_FILE:-"${EXPERIMENT_DIR}/assets/output_vocab.txt"}
PREDICTIONS_DIR="${EXPERIMENT_DIR}/trial${TRIAL_NUM}_predictions"
CHECKPOINT_PATH="${EXPERIMENT_DIR}/trial${TRIAL_NUM}/ckpt-${CHECKPOINT}"

mkdir -p ${PREDICTIONS_DIR}

# First argument is the dataset name, second is the split name
# run_inference spider dev
# run_inference yelp dev
function run_inference {
    python -m language.xsp.model.run_inference \
      --config_filepath=${EXPERIMENT_DIR}/model/model_config.json \
      --output_vocab_filepath=${OUTPUT_VOCAB_FILE} \
      --checkpoint_filepath=${CHECKPOINT_PATH} \
      --input_tfrecords=${EXPERIMENT_DIR}/tf_records/${1}_dev.tfrecords \
      --original_data_directory=${DATA_DIR}/${1} \
      --database_directory=${DATA_DIR}/databases \
      --empty_database_directory=${DATA_DIR}/databases \
      --dataset_name=${1} \
      --output_filepath=${PREDICTIONS_DIR}/${1}_dev_instructions.json \
      --inference_cache_filepath=${PREDICTIONS_DIR}/${1}_dev_inference_cache.json \
      --restored_sql_cache_filepath=${PREDICTIONS_DIR}/${1}_dev_restored_cache.json \
      --splits=${2} \
      --beam_size=1 \
      --using_abstract_sql=False \
      --spider_examples_json=${DATA_DIR}/spider/dev.json \
      --spider_tables_json=${DATA_DIR}/spider/tables.json 
}

# run_inference spider dev

# run_inference atis dev
# run_inference academic "1,2,3,4,5,6,7,8,9,0"
# # run_inference advising dev
# run_inference geoquery dev
# run_inference imdb "1,2,3,4,5,6,7,8,9,0"
# run_inference restaurants "9"
# run_inference scholar dev
# run_inference yelp "1,2,3,4,5,6,7,8,9,0"
