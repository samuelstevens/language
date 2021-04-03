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
"""Main file for training, evaluating, and tuning the NL to SQL model."""

from __future__ import absolute_import, division, print_function

import os
import pathlib
import random
import re
from typing import Any, Dict, Union

import keepsake
import tensorflow.compat.v1 as tf
from absl import app, flags

import language.xsp.evaluation.official_evaluation as official_evaluation
import language.xsp.model.inference as inference
import language.xsp.model.input_pipeline as input_pipeline
import language.xsp.model.model_builder as model_builder
import language.xsp.model.model_config as model_config
from language.xsp.evaluation import restore_from_asql

FLAGS = flags.FLAGS

flags.DEFINE_string("tf_examples_dir", "", "Path to the tensorflow examples folder.")

flags.DEFINE_string("config", "", "Path to the ModelConfig.")

flags.DEFINE_string(
    "output_vocab_filepath", "", "The location of the output vocabulary."
)
flags.DEFINE_list(
    "training_filename", "train.tfrecords@*", "Training examples TFRecords filename."
)
flags.DEFINE_string(
    "eval_filename", "eval.tfrecords@*", "Test examples TFRecords filename."
)
flags.DEFINE_string("eval_dataset_name", "", "spider, wikisql, etc.")
flags.DEFINE_integer("eval_batch_size", 8, "Batch size for eval.")
flags.DEFINE_integer("eval_beam_size", 1, "Beam width for eval.")
flags.DEFINE_string("eval_splits", "", "which splits of eval dataset to use")
flags.DEFINE_bool(
    "using_abstract_sql",
    False,
    "Whether model was trained with under-specified from clauses.",
)
flags.DEFINE_string(
    "database_directory", "", "Local directory containing the databases."
)
flags.DEFINE_string(
    "empty_database_directory", "", "Local directory containing the empty databases."
)
flags.DEFINE_string(
    "original_data_directory", "", "Directory containing the original data."
)
flags.DEFINE_string(
    "clean_output_vocab_filepath", None, "Path to the clean output vocabfile."
)

flags.DEFINE_string("model_dir", "", "Path to the model folder.")


flags.DEFINE_integer("steps_between_saves", 1000, "How many steps between saves.")

flags.DEFINE_bool(
    "use_oracle_foreign_keys",
    True,
    "Whether to use oracle foreign keys when restoring from asql.",
)

# These flags are only required if using_abstract_sql is True.
# TODO(samuelstevens): Better method for handling other datasets.
flags.DEFINE_string("spider_examples_json", "", "Path to Spider json examples")
flags.DEFINE_string("spider_tables_json", "", "Path to Spider json tables.")


KEEP_CHECKPOINTS_MAX = 5


class ValidationHook(tf.train.SessionRunHook):
    _config: inference.Config
    _validation_query_cache: Dict[str, Any]

    def __init__(
        self, experiment, config: inference.Config, every_n_steps=None,
    ):
        self._iter_count = 0
        self._config = config
        self._should_trigger = False
        self._experiment = experiment
        self._validation_query_cache = {}
        self._timer = tf.train.SecondOrStepTimer(None, every_n_steps)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

    def after_run(self, run_context, run_values) -> None:
        if not self._should_trigger:
            self._iter_count += 1
            return

        examples = inference.load_tf_examples(
            os.path.join(FLAGS.tf_examples_dir, FLAGS.eval_filename)
        )
        random.shuffle(examples)

        checkpoint = self.get_latest_checkpoint(FLAGS.model_dir)

        predictions = inference.inference(examples, checkpoint, self._config)

        if FLAGS.using_abstract_sql:
            is_spider = "spider" == FLAGS.eval_dataset_name.lower()

            michigan_schema = None
            if not is_spider:
                michigan_schema = inference.load_schema_obj(
                    FLAGS.eval_dataset_name, FLAGS.original_data_directory
                )

            predictions = restore_from_asql.restore_from_clauses(
                predictions,
                spider_examples_json=FLAGS.spider_examples_json if is_spider else "",
                spider_tables_json=FLAGS.spider_tables_json if is_spider else "",
                michigan_schema=michigan_schema,
                dataset_name=FLAGS.eval_dataset_name,
                use_oracle_foreign_keys=FLAGS.use_oracle_foreign_keys,
            )

        # Load the database tables.
        schema_obj = inference.load_schema_obj(
            FLAGS.eval_dataset_name, FLAGS.original_data_directory
        )

        # Now match with the original data and save
        examples_to_execute = inference.match_and_save(
            self._config, predictions, schema_obj
        )

        # Only update cache when it's empty
        should_update_cache = len(self._validation_query_cache) == 0

        results, validation_query_cache = official_evaluation.execute_predictions(
            examples_to_execute,
            self._validation_query_cache,
            "scholar" not in FLAGS.eval_dataset_name.lower(),
            False,
            should_update_cache,
        )

        metrics = official_evaluation.aggregate_metrics(results)

        self._experiment.checkpoint(
            step=self._iter_count,
            metrics={"eval_execution_f1": metrics.execution_f1},
            primary_metric=("eval_execution_f1", "maximize"),
        )

        self._timer.update_last_triggered_step(self._iter_count)
        self._iter_count += 1

    def get_latest_checkpoint(self, model_dir: Union[str, pathlib.Path]) -> str:
        if isinstance(model_dir, str):
            model_dir = pathlib.Path(model_dir)

        assert os.path.isfile(model_dir / "checkpoint")

        pattern = re.compile('all_model_checkpoint_paths: "ckpt-(.*)"')
        latest = 0
        with open(model_dir / "checkpoint") as checkpoint_file:
            for line in checkpoint_file:
                if not line.startswith("all_model_checkpoint_paths"):
                    continue

                pattern_match = pattern.match(line)
                if not pattern_match:
                    continue
                checkpoint_num = pattern_match.group(1)

                if int(checkpoint_num) > latest:
                    latest = int(checkpoint_num)

        return str(model_dir / f"ckpt-{latest}")


def global_seed(seed: int) -> None:
    tf.random.set_random_seed(seed)


def main(unused_argv: Any) -> None:
    tf.logging.info("Saving model saves and results to " + FLAGS.model_dir)

    global_seed(42)

    config = model_config.load_config(FLAGS.config)

    tf.logging.info("Training with train filenames: " + str(FLAGS.training_filename))

    # Training allows noisy examples so do not use clean output vocab
    model_fn = model_builder.build_model_fn(
        config, FLAGS.output_vocab_filepath, clean_output_vocab_path=""
    )

    # for keepsake CLI (helps track experiment results)
    experiment = keepsake.init(
        params={
            "learning_rate": config.training_options.optimizer_learning_rate,
            "batch_size": config.training_options.batch_size,
            "training_steps": config.training_options.training_steps,
            "eval_batch_size": FLAGS.eval_batch_size,
            "training_data": FLAGS.training_filename,
            "eval_data": FLAGS.eval_filename,
        },
    )

    train_input_fn = input_pipeline.create_training_input_fn(
        config,
        FLAGS.tf_examples_dir,
        [name for name in FLAGS.training_filename if name],
    )

    inference_config = inference.Config(
        FLAGS.eval_dataset_name,
        FLAGS.eval_splits.split(","),
        FLAGS.output_vocab_filepath,
        FLAGS.clean_output_vocab_filepath,
        FLAGS.eval_beam_size,
        FLAGS.using_abstract_sql,
        FLAGS.database_directory,
        FLAGS.empty_database_directory,
        FLAGS.original_data_directory,
        model_config.load_config(FLAGS.config),
    )

    validation_hook = ValidationHook(
        every_n_steps=FLAGS.steps_between_saves,
        experiment=experiment,
        config=inference_config,
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        tf_random_seed=42,
        save_summary_steps=1,
        save_checkpoints_steps=FLAGS.steps_between_saves,
        keep_checkpoint_max=KEEP_CHECKPOINTS_MAX,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params={}, model_dir=FLAGS.model_dir, config=run_config
    )

    estimator.train(
        input_fn=train_input_fn,
        max_steps=config.training_options.training_steps,
        hooks=[validation_hook],
    )

    # experiment.checkpoint(
    #     step=global_step,
    #     metrics={"train_loss": train_loss, "eval_execution_f1": metrics.execution_f1,},
    #     primary_metric=("eval_execution_f1", "maximize"),
    # )

    # endregion


if __name__ == "__main__":
    app.run(main)
