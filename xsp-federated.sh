#!/usr/bin/env bash

python -m language.xsp.training.train_model \
  --do_train \
  --tf_examples_dir=xsp_experiment_run/tf_records/ \
  --config=xsp_experiment_run/model/model_config.json \
  --output_vocab=xsp_experiment_run/assets/output_vocab.txt \
  --training_filename=spider_train.tfrecords,wikisql_train.tfrecords \
  --eval_filename=spider_dev.tfrecords \
  --model_dir=xsp_experiment_run/experiment_trial_0 \
  --eval_batch_size=1 \
  --steps_between_saves=1000
