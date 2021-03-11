#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} 

DATA_DIR=${DATA_DIR:-"data"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"xsp_experiment_run"}
OUTPUT_VOCAB_FILE=${OUTPUT_VOCAB_FILE:-"${EXPERIMENT_DIR}/assets/output_vocab.txt"}
PREDICTIONS_DIR=${PREDICTIONS_DIR:-"${EXPERIMENT_DIR}/predictions"}

python -m language.xsp.training.train_model \
  --tf_examples_dir=${EXPERIMENT_DIR}/tf_records/ \
  --config=${EXPERIMENT_DIR}/model/model_config.json \
  --output_vocab=${OUTPUT_VOCAB_FILE} \
  --training_filename=spider_train.tfrecords,wikisql_train.tfrecords \
  --eval_filename=spider_dev.tfrecords \
  --model_dir=${EXPERIMENT_DIR}/trial${TRIAL_NUM} \
  --eval_batch_size=8 \
  --steps_between_saves=5000 \
  --do_train \
  --do_eval 
