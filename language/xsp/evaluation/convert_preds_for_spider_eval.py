# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
r"""Converts a predictions json file to format expected for Spider evaluation.

The input to this binary is the predictions file generated by
`run_inference.py`, which contains newline separated json dictionaries with
model predictions and scores. This file currently selects the one-best
score.

The output is a newline separated text file containing the one-best predictions
and the associated database name.

Example usage:

${PATH_TO_BINARY} \
  --spider_examples_json=${SPIDER_DIR}/spider/dev.json \
  --input_path=${INPUT} \
  --output_path=${SPIDER_PREDICTIONS}

python ${SPIDER_DIR}/spider-master/evaluation.py \
  --gold "${SPIDER_DIR}/spider/dev_gold.sql" \
  --pred "${SPIDER_PREDICTIONS}" \
  --etype match \
  --db "${SPIDER_DIR}/spider/database" \
  --table "${SPIDER_DIR}/spider/tables.json"
"""

from __future__ import absolute_import, division, print_function

import json
import sqlite3

import tensorflow.compat.v1 as tf
from absl import app, flags

from language.xsp.evaluation import official_evaluation

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Path to predictions.")
flags.DEFINE_string("output_path", "", "Path to write Spider-formatted predictions.")
flags.DEFINE_string("spider_examples_json", "", "Path to Spider json examples")
flags.DEFINE_bool(
    "use_executable_sql_only",
    True,
    "If true, run the query on the empty databases" "to filter un-executable queries",
)


def _load_json(filepath):
    with tf.gfile.Open(filepath) as json_file:
        return json.load(json_file)


def _load_predictions_dicts(filepath):
    """Returns list of predictions dicts."""
    predictions_dicts = json.load(open(filepath))
    return predictions_dicts


def _utterance_to_one_best_sql_map(predictions_dicts):
    """Get map of utterance to best prediction."""
    # Assume sorted in descending order by score.
    utterance_to_one_best_sql_map = {}
    for prediction in predictions_dicts:
        if not prediction["predictions"]:
            print("No predictions for example: %s" % prediction)
            continue

        utterance = prediction["utterance"]
        paired_preds_and_scores = zip(prediction["predictions"], prediction["scores"])
        sorted_by_scores = sorted(
            paired_preds_and_scores, key=lambda x: x[1], reverse=True
        )

        if FLAGS.use_executable_sql_only:
            empty_path = prediction["empty_database_path"]
            try:
                empty_conn = sqlite3.connect(empty_path)
                empty_conn.text_factory = str
            except sqlite3.OperationalError as e:
                print(e)
                print(empty_path)
                exit()

            cursor = empty_conn.cursor()
            best_prediction = None
            for _, (pred, _) in enumerate(sorted_by_scores):
                # Try predicting
                print("Trying to execute query:\n\t" + pred)
                print("... on empty database")
                temp_exception_str = official_evaluation.try_executing_query(
                    pred, cursor, case_sensitive=True
                )[1]

                if temp_exception_str:
                    if temp_exception_str == "timeout":
                        # Technically, this query didn't have a syntax problem, so
                        # continue and set this as the best prediction.
                        best_prediction = pred
                        break
                else:
                    best_prediction = pred
                    break

            one_best_prediction = best_prediction
        else:
            one_best_prediction = sorted_by_scores[0][0]
        utterance_to_one_best_sql_map[utterance] = one_best_prediction
    return utterance_to_one_best_sql_map


def write_predictions(input_path, output_path, spider_examples_json):
    """Writes one-best predictions in Spider-evaluation compatible format."""
    predictions_dicts = _load_predictions_dicts(input_path)
    utterance_to_one_best_sql_map = _utterance_to_one_best_sql_map(predictions_dicts)

    examples = _load_json(spider_examples_json)
    with tf.gfile.Open(output_path, "w") as output_file:
        for example in examples:
            utterance = " ".join(example["question_toks"])
            db_id = example["db_id"]
            if utterance not in utterance_to_one_best_sql_map:
                print("No prediction for utterance: %s" % utterance)
                # Write a line with dummy output.
                output_file.write("SKIP\t%s\n" % db_id)
            else:
                prediction = utterance_to_one_best_sql_map[utterance]
                output_file.write("%s\t%s\n" % (prediction, db_id))


def main(unused_argv):
    write_predictions(FLAGS.input_path, FLAGS.output_path, FLAGS.spider_examples_json)


if __name__ == "__main__":
    app.run(main)
