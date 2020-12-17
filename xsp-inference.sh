#!/usr/bin/env bash

python -m language.xsp.model.run_inference \
  --output_vocab_filepath=xsp_experiment_run/assets/output_vocab.txt \
  --checkpoint_filepath=xsp_experiment_run/experiment_trial_0/model.ckpt-3000 \
  --config_filepath=xsp_experiment_run/model/model_config.json \
  --input=xsp_experiment_run/tf_records/spider_dev.tfrecords \
  --data_filepath=data/spider \
  --database_filepath=data/spider/database \
  --output=predictions/final_dataset_predictions.json \
  --splits=dev \
  --dataset_name=spider \
  --beam_size=1 \
  --predictions_path=predictions/spider-dev-predictions.json
