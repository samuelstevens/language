#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR=${EXPERIMENT_DIR:-"xsp_experiment_run"}
DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_VOCAB_FILE=${OUTPUT_VOCAB_FILE:-"${EXPERIMENT_DIR}/assets/output_vocab.txt"}

# 1. Convert raw data to JSON
function raw_to_json {
    # atis dev
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=atis \
        --input_filepath=${DATA_DIR}/atis/ \
        --splits=dev \
        --output_filepath=${EXPERIMENT_DIR}/examples/atis_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # geoquery dev
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=geoquery \
        --input_filepath=${DATA_DIR}/geoquery/ \
        --splits=dev \
        --output_filepath=${EXPERIMENT_DIR}/examples/geoquery_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # advising dev
    # echo "Skipping advising dataset for now"
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=advising \
        --input_filepath=${DATA_DIR}/advising/ \
        --splits=dev \
        --output_filepath=${EXPERIMENT_DIR}/examples/advising_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # scholar dev
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=scholar \
        --input_filepath=${DATA_DIR}/scholar/ \
        --splits=dev \
        --output_filepath=${EXPERIMENT_DIR}/examples/scholar_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # restaurants all folds
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=restaurants \
        --input_filepath=${DATA_DIR}/restaurants/ \
        --splits=0,1,2,3,4,5,6,7,8,9 \
        --output_filepath=${EXPERIMENT_DIR}/examples/restaurants_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # imdb all folds
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=imdb \
        --input_filepath=${DATA_DIR}/imdb/ \
        --splits=0,1,2,3,4,5,6,7,8,9 \
        --output_filepath=${EXPERIMENT_DIR}/examples/imdb_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # academic all folds
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=academic \
        --input_filepath=${DATA_DIR}/academic/ \
        --splits=0,1,2,3,4,5,6,7,8,9 \
        --output_filepath=${EXPERIMENT_DIR}/examples/academic_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # yelp all folds
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=yelp \
        --input_filepath=${DATA_DIR}/yelp/ \
        --splits=0,1,2,3,4,5,6,7,8,9 \
        --output_filepath=${EXPERIMENT_DIR}/examples/yelp_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # spider training
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=spider \
        --input_filepath=${DATA_DIR}/spider/ \
        --splits=train \
        --output_filepath=${EXPERIMENT_DIR}/examples/spider_train.json \
        --generate_sql=True \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # wikisql training
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=wikisql \
        --input_filepath=${DATA_DIR}/wikisql/ \
        --splits=train \
        --output_filepath=${EXPERIMENT_DIR}/examples/wikisql_train.json \
        --generate_sql=True \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt

    # spider dev
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=spider \
        --input_filepath=${DATA_DIR}/spider/ \
        --splits=dev \
        --output_filepath=${EXPERIMENT_DIR}/examples/spider_dev.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt
}

function output_vocab {
    # 2. Create output vocabulary from training data
    python -m language.xsp.data_preprocessing.create_vocabularies \
        --data_dir=${EXPERIMENT_DIR}/examples/ \
        --input_filenames=spider_train.json,wikisql_train.json \
        --output_path=${OUTPUT_VOCAB_FILE}
}

function json_to_tfrecords {
    # 3. Convert to TFRecords

    # xsp dev
    echo "Add advising_dev.json when it's fixed"
    python -m language.xsp.data_preprocessing.convert_to_tfrecords \
        --examples_dir=${EXPERIMENT_DIR}/examples/ \
        --filenames=atis_dev.json,academic_dev.json,imdb_dev.json,geoquery_dev.json,scholar_dev.json,restaurants_dev.json,yelp_dev.json \
        --output_vocab=${OUTPUT_VOCAB_FILE} \
        --permute=False \
        --config=${EXPERIMENT_DIR}/model/model_config.json \
        --tf_examples_dir=${EXPERIMENT_DIR}/tf_records

    # training data (spider and wikisql)
    python -m language.xsp.data_preprocessing.convert_to_tfrecords \
        --examples_dir=${EXPERIMENT_DIR}/examples/ \
        --filenames=spider_train.json,wikisql_train.json \
        --output_vocab=${OUTPUT_VOCAB_FILE} \
        --generate_output=True \
        --permute=True \
        --num_spider_repeats=7 \
        --config=${EXPERIMENT_DIR}/model/model_config.json \
        --tf_examples_dir=${EXPERIMENT_DIR}/tf_records

    # spider dev
    python -m language.xsp.data_preprocessing.convert_to_tfrecords \
        --examples_dir=${EXPERIMENT_DIR}/examples/ \
        --filenames=spider_dev.json \
        --output_vocab=${OUTPUT_VOCAB_FILE} \
        --permute=False \
        --config=${EXPERIMENT_DIR}/model/model_config.json \
        --tf_examples_dir=${EXPERIMENT_DIR}/tf_records
}

raw_to_json
output_vocab
json_to_tfrecords
