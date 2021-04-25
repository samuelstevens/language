#!/usr/bin/env bash

set -euo pipefail
set -x

# strip trailing slashes
EXPERIMENT_DIR=$(realpath --canonicalize-missing ${EXPERIMENT_DIR})

DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_VOCAB_FILE=${OUTPUT_VOCAB_FILE:-"${EXPERIMENT_DIR}/assets/output_vocab.txt"}

function raw_to_eval_json {
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=${1} \
        --input_filepath=${DATA_DIR}/${1}/ \
        --splits=${2} \
        --output_filepath=${EXPERIMENT_DIR}/examples/${1}_${3}.json \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt
}

function raw_to_training_json {
    python -m language.xsp.data_preprocessing.convert_to_examples \
        --dataset_name=${1} \
        --input_filepath=${DATA_DIR}/${1}/ \
        --splits=${2} \
        --output_filepath=${EXPERIMENT_DIR}/examples/${1}_${3}.json \
        --generate_sql=True \
        --tokenizer_vocabulary=${EXPERIMENT_DIR}/assets/input_bert_vocabulary.txt \
        --abstract_sql=${4}
}

function join_by_char {
    # https://dev.to/meleu/how-to-join-array-elements-in-a-bash-script-303a
    local IFS="$1"
    shift
    echo "$*"
}

function output_vocab {
    input_filenames=()
    for dataset in $@; do
        input_filenames+=(${dataset}_train.json)
    done
    input_filenames=$(join_by_char , ${input_filenames})
    # 2. Create output vocabulary from training data
    python -m language.xsp.data_preprocessing.create_vocabularies \
        --data_dir=${EXPERIMENT_DIR}/examples/ \
        --input_filenames=${input_filenames} \
        --output_path=${OUTPUT_VOCAB_FILE}
}

function json_to_eval_tf_records {
    python -m language.xsp.data_preprocessing.convert_to_tfrecords \
        --examples_dir=${EXPERIMENT_DIR}/examples/ \
        --filenames=${1}_${2}.json \
        --output_vocab=${OUTPUT_VOCAB_FILE} \
        --permute=False \
        --config=${EXPERIMENT_DIR}/model/model_config.json \
        --tf_examples_dir=${EXPERIMENT_DIR}/tf_records

}

function json_to_training_tf_records {
    # training data (spider and wikisql)
    python -m language.xsp.data_preprocessing.convert_to_tfrecords \
        --examples_dir=${EXPERIMENT_DIR}/examples/ \
        --filenames=${1}_${2}.json \
        --output_vocab=${OUTPUT_VOCAB_FILE} \
        --generate_output=True \
        --permute=True \
        --num_spider_repeats=7 \
        --config=${EXPERIMENT_DIR}/model/model_config.json \
        --tf_examples_dir=${EXPERIMENT_DIR}/tf_records
}

# ssp dev sets
# raw_to_eval_json atis dev dev
# raw_to_eval_json geoquery dev dev
# raw_to_eval_json scholar dev dev
# raw_to_eval_json imdb "9" dev
# raw_to_eval_json restaurants "9" dev
# raw_to_eval_json yelp "0,1,2,3,4,5,6,7,8,9" dev
# raw_to_eval_json academic "0,1,2,3,4,5,6,7,8,9" dev

# # xsp training
# raw_to_training_json wikisql train train False
# raw_to_training_json spider train train False

# raw_to_eval_json spider dev dev

# ssp train sets
# raw_to_training_json atis train train False
# raw_to_training_json scholar train train False
# raw_to_training_json geoquery train train False
# raw_to_training_json restaurants "0,1,2,3,4,5,6,7,8" train False
# raw_to_training_json imdb "0,1,2,3,4,5,6,7,8" train False

# output_vocab imbd

# json_to_eval_tf_records atis dev
# json_to_eval_tf_records academic dev
# json_to_eval_tf_records advising dev
# json_to_eval_tf_records imdb dev
# json_to_eval_tf_records geoquery dev
# json_to_eval_tf_records scholar dev
# json_to_eval_tf_records geoquery dev
# json_to_eval_tf_records yelp dev

# json_to_eval_tf_records spider dev

# json_to_training_tf_records spider train
# json_to_training_tf_records wikisql train
# json_to_training_tf_records scholar train
# json_to_training_tf_records geoquery train
# json_to_training_tf_records atis train
# json_to_training_tf_records imdb train
