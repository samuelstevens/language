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
"""Utilities for restoring under-specified FROM clause at inference time."""

from __future__ import absolute_import, division, print_function

import json
from typing import List, Sequence

import tensorflow.compat.v1 as tf
import tqdm

from language.xsp.data_preprocessing import abstract_sql, abstract_sql_converters
from language.xsp.model.inference import Prediction


def _load_json(filepath):
    with tf.gfile.Open(filepath) as json_file:
        return json.load(json_file)


def _load_predictions_dicts(filepath):
    """Returns list of predictions dicts."""
    predictions_dicts = []
    # Check if input is sharded.
    if filepath.endswith("*"):
        for data_file in tf.gfile.Glob(filepath):
            print("Loading from file %s." % data_file)
            with tf.gfile.Open(data_file) as infile:
                for line in infile:
                    if line:
                        predictions_dict = json.loads(line)
                        predictions_dicts.append(predictions_dict)

    else:
        print("Loading from file %s." % filepath)
        with tf.gfile.Open(filepath) as infile:
            for line in infile:
                if line:
                    predictions_dict = json.loads(line)
                    predictions_dicts.append(predictions_dict)
    print("Loaded %s prediction dicts." % len(predictions_dicts))
    return predictions_dicts


def _utterance_to_db_map(spider_examples_json, spider_tables_json):
    """Returns map of utterances to Spider db json object."""
    print("Loading tables from %s" % spider_examples_json)
    tables = _load_json(spider_tables_json)
    db_id_to_db_map = {db["db_id"]: db for db in tables}

    utterance_to_db_map = {}
    examples = _load_json(spider_examples_json)
    for example in examples:
        db = db_id_to_db_map[example["db_id"]]
        utterance = " ".join(example["question_toks"])
        if utterance in utterance_to_db_map:
            raise ValueError("Utterance %s already in map." % utterance)
        utterance_to_db_map[utterance] = db

    return utterance_to_db_map


def _get_restored_predictions(
    predictions_dict,
    utterance_to_db_map=None,
    schema=None,
    dataset_name=None,
    use_oracle_foreign_keys=False,
):
    """
    Returns new predictions dict with FROM clauses restored.
    """
    utterance = predictions_dict["utterance"]
    if utterance_to_db_map:
        db = utterance_to_db_map[utterance]
        foreign_keys = abstract_sql_converters.spider_db_to_foreign_key_tuples(db)
        table_schemas = abstract_sql_converters.spider_db_to_table_tuples(db)

    else:
        if use_oracle_foreign_keys:
            foreign_keys = abstract_sql_converters.michigan_db_to_foreign_key_tuples_orcale(
                dataset_name
            )
        else:
            foreign_keys = abstract_sql_converters.michigan_db_to_foreign_key_tuples(
                schema
            )
        table_schemas = abstract_sql_converters.michigan_db_to_table_tuples(schema)

    restored_predictions = []
    restored_scores = []
    for prediction, score in zip(
        predictions_dict["predictions"], predictions_dict["scores"]
    ):
        # Some predictions have repeated single quotes around values.
        prediction = prediction.replace("''", "'")

        try:
            restored_prediction = abstract_sql_converters.restore_predicted_sql(
                prediction, table_schemas, foreign_keys
            )
            restored_predictions.append(restored_prediction)
            restored_scores.append(score)
        except abstract_sql.UnsupportedSqlError as e:
            # Remove predictions that fail conversion.
            print("For query %s" % prediction)
            print("Unsupport Error: " + str(e))
        except abstract_sql.ParseError as e:
            print("For query %s" % prediction)
            print("Parse Error: " + str(e))

    restored_predictions_dict = {
        "utterance": utterance,
        "predictions": restored_predictions,
        "scores": restored_scores,
    }
    return restored_predictions_dict


def restore_from_clauses(
    predictions: Sequence[Prediction],
    spider_examples_json="",
    spider_tables_json="",
    michigan_schema=None,
    dataset_name=None,
    use_oracle_foreign_keys: bool = False,
) -> List[Prediction]:
    """
    Loads an original dataset and matches with a predictions file.

    The input and output is a text file containing model predictions.
    This is a newline-separated file of json-encoded predictions dictionary.
    Each dictionary contains keys:
    - `utterance` - The natural language query.
    - `predictions` - List of predicted SQL string.
    - `scores` - List of predicted scores.

    Args:
        input_path: Path to input predictions.
        output_path: Path to write output predictions.
        spider_examples_json: Path to Spider examples.
        spider_tables_json: Path to Spider tables.
        michigan_schema: A Michigan schema object (list of tables, each with a list
          of columns and their types).
        dataset_name: Name of Michigan dataset if using oracle foreign keys.
        use_oracle_foreign_keys: Whether to use oracle foreign keys for Michigan.
    """
    # Create map of utterances to schemas.
    utterance_to_db_map = None
    if spider_examples_json:
        utterance_to_db_map = _utterance_to_db_map(
            spider_examples_json, spider_tables_json
        )
    elif not michigan_schema:
        raise ValueError(
            "Must provide either a filepath to Spider schema specification, "
            "or a Michigan schema object."
        )

    # Restore from clauses.
    restored_predictions = []
    for prediction in tqdm.tqdm(predictions):
        restored_predictions.append(
            _get_restored_predictions(
                prediction,
                utterance_to_db_map,
                michigan_schema,
                dataset_name,
                use_oracle_foreign_keys,
            )
        )

    return restored_predictions
