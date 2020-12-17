#!/usr/bin/env bash

python -m language.xsp.evaluation.official_evaluation \
  --predictions_filepath=predictions/final_dataset_predictions.json \
  --output_filepath=predictions/dataset_predictions.txt \
  --cache_filepath=dataset_cache.json \
  --verbose=False \
  --update_cache=True
