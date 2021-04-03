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
import statistics
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

flags.DEFINE_bool("do_train", False, "Whether to do training.")

flags.DEFINE_bool("do_eval", False, "Whether to do evaluation continuously.")

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


def delete_checkpoint(model_dir: Union[str, pathlib.Path], num: int) -> None:
    if isinstance(model_dir, str):
        model_dir = pathlib.Path(model_dir)

    for checkpoint_file in (model_dir / f"ckpt-{num}").glob("*"):
        os.remove(checkpoint_file)


def global_seed(seed: int) -> None:
    tf.random.set_random_seed(seed)


def main(unused_argv: Any) -> None:
    tf.logging.info("Saving model saves and results to " + FLAGS.model_dir)

    global_seed(42)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train`, `do_eval` must be True.")

    config = model_config.load_config(FLAGS.config)

    if FLAGS.do_train:
        tf.logging.info(
            "Training with train filenames: " + str(FLAGS.training_filename)
        )

    # Training allows noisy examples so do not use clean output vocab
    model_fn = model_builder.build_model_fn(
        config, FLAGS.output_vocab_filepath, clean_output_vocab_path=""
    )

    # region training
    if FLAGS.do_train:
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

        train_features, train_labels = train_input_fn()
        train_model = model_fn(
            train_features, train_labels, tf.estimator.ModeKeys.TRAIN
        )

        tf.get_variable_scope().reuse_variables()

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

        # each checkpoint is ~1.3 GB, so 20 is ~25GB.
        # TODO(samuelstevens): This can be turned down once I know a rough estimate for how many steps the model should train (based on validation loss).
        saver = tf.train.Saver(max_to_keep=20)

        global_step = 0
        checkpoint = f"{FLAGS.model_dir}/ckpt-{global_step}"

        validation_query_cache: Dict[str, Any] = {}

        with tf.Session() as init_sess:
            init_sess.run(tf.global_variables_initializer())
            saver.save(init_sess, checkpoint)

        while global_step < config.training_options.training_steps:
            # region training loop
            with tf.Session() as train_sess:
                tf.logging.info(
                    "Training from step %s to step %s",
                    global_step,
                    global_step + FLAGS.steps_between_saves,
                )
                saver.restore(train_sess, checkpoint)

                train_losses = []

                for step in range(FLAGS.steps_between_saves):
                    _, train_loss = train_sess.run(
                        [train_model.train_op, train_model.loss]
                    )

                    train_losses.append(train_loss)

                    if step % 100 == 0:
                        tf.logging.info(
                            "Step %s's training loss: %s",
                            global_step + step,
                            train_loss,
                        )

                train_loss = statistics.mean(train_losses)

                global_step += FLAGS.steps_between_saves
                checkpoint = f"{FLAGS.model_dir}/ckpt-{global_step}"
                saver.save(train_sess, checkpoint)
            # endregion

            # region eval loop
            tf.logging.info("Evaluating checkpoint %s", checkpoint)

            tf.logging.info("Running inference on %s", FLAGS.eval_filename)

            examples = inference.load_tf_examples(
                os.path.join(FLAGS.tf_examples_dir, FLAGS.eval_filename)
            )
            random.shuffle(examples)

            predictions = inference.inference(examples, checkpoint, inference_config,)

            if FLAGS.using_abstract_sql:
                is_spider = "spider" == FLAGS.eval_dataset_name.lower()

                michigan_schema = None
                if not is_spider:
                    michigan_schema = inference.load_schema_obj(
                        FLAGS.eval_dataset_name, FLAGS.original_data_directory
                    )

                predictions = restore_from_asql.restore_from_clauses(
                    predictions,
                    spider_examples_json=FLAGS.spider_examples_json
                    if is_spider
                    else "",
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
                inference_config, predictions, schema_obj
            )

            # Only update cache when it's empty
            should_update_cache = len(validation_query_cache) == 0

            results, validation_query_cache = official_evaluation.execute_predictions(
                examples_to_execute,
                validation_query_cache,
                "scholar" not in FLAGS.eval_dataset_name.lower(),
                False,
                should_update_cache,
            )

            metrics = official_evaluation.aggregate_metrics(results)
            tf.logging.info(
                "Validation Results:\n\tExecution F1: %s", metrics.execution_f1
            )
            # endregion

            experiment.checkpoint(
                step=global_step,
                metrics={
                    "train_loss": train_loss,
                    "eval_execution_f1": metrics.execution_f1,
                },
                primary_metric=("eval_execution_f1", "maximize"),
            )

    # endregion


if __name__ == "__main__":
    app.run(main)
