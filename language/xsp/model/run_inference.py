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
"""Runs inference on an XSP model."""
from __future__ import absolute_import, division, print_function

import json
import os
from typing import Any, Callable, TypeVar

from absl import app, flags

import language.xsp.evaluation.restore_from_asql as restore_from_asql
import language.xsp.model.inference as inference
import language.xsp.model.model_config as model_config

T = TypeVar("T")

FLAGS = flags.FLAGS

flags.DEFINE_string("config_filepath", "", "The location of the model config.")

flags.DEFINE_string(
    "output_vocab_filepath", "", "The location of the output vocabulary."
)

flags.DEFINE_string(
    "clean_output_vocab_filepath", None, "Path to the clean output vocabfile."
)

flags.DEFINE_string("checkpoint_filepath", "", "The location of the checkpoint.")

flags.DEFINE_string(
    "input_tfrecords", "", "TFRecords file containing TFExamples to evaluate."
)

flags.DEFINE_string(
    "original_data_directory", "", "Directory containing the original data."
)
flags.DEFINE_string(
    "database_directory", "", "Local directory containing the databases."
)
flags.DEFINE_string(
    "empty_database_directory", "", "Local directory containing the empty databases."
)
flags.DEFINE_string("output_filepath", "", "Path for the output json file.")
flags.DEFINE_string("inference_cache_filepath", "", "Path to cache inference results.")
flags.DEFINE_string(
    "restored_sql_cache_filepath",
    "",
    "Path to cache predictions after restoring FROM clauses",
)
flags.DEFINE_list("splits", [], "The splits to run with.")
flags.DEFINE_string("dataset_name", "", "The name of the dataset being processed.")

flags.DEFINE_integer("beam_size", 1, "The size of the beam to predict.")

flags.DEFINE_bool(
    "using_abstract_sql",
    False,
    "Whether model was trained with under-specified from clauses.",
)
flags.DEFINE_bool(
    "use_oracle_foreign_keys",
    True,
    "Whether to use oracle foreign keys when restoring from asql.",
)
# These flags are only required if using_abstract_sql is True.
# TODO(petershaw): Better method for handling other datasets.
flags.DEFINE_string("spider_examples_json", "", "Path to Spider json examples")
flags.DEFINE_string("spider_tables_json", "", "Path to Spider json tables.")


def cached_fn_call(fn: Callable[[], T], filepath: str) -> T:
    """
    Caches a function call on disk. If the file exists, use that. Otherwise, call the function and save to disk.
    """
    if not os.path.isfile(filepath):
        result = fn()
        with open(filepath, "w") as file:
            file.write(json.dumps(result) + "\n")
    else:
        with open(filepath) as file:
            result = json.load(file)

    return result


def inference_wrapper(sharded: bool = False):
    """
    Wrapper for running inference.
    """

    assert (
        FLAGS.output_filepath
    ), "You must provide a filepath to write evaluation instructions to."
    assert FLAGS.output_filepath.endswith(".json"), "Output file must be .json"

    assert isinstance(FLAGS.splits, list), f"Expected a list; got {FLAGS.splits}"
    assert len(FLAGS.splits) > 0, f"Expected a non-empty list; got {FLAGS.splits}"

    config = inference.Config(
        dataset_name=FLAGS.dataset_name,
        splits=FLAGS.splits,
        output_vocab_filepath=FLAGS.output_vocab_filepath,
        clean_output_vocab_filepath=FLAGS.clean_output_vocab_filepath,
        beam_size=FLAGS.beam_size,
        using_abstract_sql=FLAGS.using_abstract_sql,
        database_directory=FLAGS.database_directory,
        empty_database_directory=FLAGS.empty_database_directory,
        original_data_directory=FLAGS.original_data_directory,
        model_config=model_config.load_config(FLAGS.config_filepath),
    )

    examples = inference.load_tf_examples(FLAGS.input_tfrecords)

    print(f"Performing inference on {len(examples)} examples.")
    predictions = cached_fn_call(
        lambda: inference.inference(examples, FLAGS.checkpoint_filepath, config),
        FLAGS.inference_cache_filepath,
    )

    # If using Abstract SQL, need to restore under-specified FROM clauses output above.
    if FLAGS.using_abstract_sql:
        is_spider = FLAGS.dataset_name.lower() == "spider"

        michigan_schema = None
        if not is_spider:
            michigan_schema = inference.load_schema_obj(
                FLAGS.dataset_name, FLAGS.original_data_directory
            )
        print(f"Restoring FROM clauses for {len(predictions)} predictions")
        predictions = cached_fn_call(
            lambda: restore_from_asql.restore_from_clauses(
                predictions,
                spider_examples_json=FLAGS.spider_examples_json if is_spider else "",
                spider_tables_json=FLAGS.spider_tables_json if is_spider else "",
                michigan_schema=michigan_schema,
                dataset_name=FLAGS.dataset_name,
                use_oracle_foreign_keys=FLAGS.use_oracle_foreign_keys,
            ),
            FLAGS.restored_sql_cache_filepath,
        )

    # Load the database tables.
    schema_obj = inference.load_schema_obj(
        FLAGS.dataset_name, FLAGS.original_data_directory
    )

    # Now match with the original data and save
    cached_fn_call(
        lambda: inference.match_and_save(config, predictions, schema_obj),
        FLAGS.output_filepath,
    )


def main(unused_argv: Any) -> None:
    inference_wrapper()


if __name__ == "__main__":
    app.run(main)
