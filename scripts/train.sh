#!/usr/bin/env bash

set -euo pipefail

# force use to set CUDA_VISIBLE_DEVICES before running this script.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} 

# strip trailing slashes
EXPERIMENT_DIR=$(realpath --canonicalize-missing ${EXPERIMENT_DIR})

DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_VOCAB_FILE=${OUTPUT_VOCAB_FILE:-"${EXPERIMENT_DIR}/assets/output_vocab.txt"}
PREDICTIONS_DIR=${PREDICTIONS_DIR:-"${EXPERIMENT_DIR}/predictions"}


python -m language.xsp.training.train_model \
  --tf_examples_dir=${EXPERIMENT_DIR}/tf_records/ \
  --config=${EXPERIMENT_DIR}/model/model_config.json \
  --output_vocab_filepath=${OUTPUT_VOCAB_FILE} \
  --training_filename=spider_train.tfrecords,wikisql_train.tfrecords \
  --eval_filename=spider_dev.tfrecords \
  --model_dir=${EXPERIMENT_DIR}/trial${TRIAL_NUM} \
  --eval_batch_size=8 \
  --steps_between_saves=10000 \
  --eval_dataset_name=spider \
  --eval_splits=dev \
  --eval_beam_size=1 \
  --using_abstract_sql=True \
  --database_directory=${DATA_DIR}/databases \
  --empty_database_directory=${DATA_DIR}/empty_databases \
  --original_data_directory=${DATA_DIR}/spider \
  --spider_examples_json=${DATA_DIR}/spider/dev.json \
  --spider_tables_json=${DATA_DIR}/spider/tables.json \
  --do_train \
  --do_eval 
