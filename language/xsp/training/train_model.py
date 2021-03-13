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
import re

import keepsake
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.gfile as gfile
from absl import app, flags

import language.xsp.model.input_pipeline as input_pipeline
import language.xsp.model.model_builder as model_builder
import language.xsp.model.model_config as model_config

FLAGS = flags.FLAGS

flags.DEFINE_bool("do_train", False, "Whether to do training.")

flags.DEFINE_bool("do_eval", False, "Whether to do evaluation continuously.")

flags.DEFINE_string("tf_examples_dir", "", "Path to the tensorflow examples folder.")

flags.DEFINE_string("config", "", "Path to the ModelConfig.")

flags.DEFINE_string("output_vocab", "", "Path to the output vocab file.")

flags.DEFINE_list(
    "training_filename", "train.tfrecords@*", "Training examples TFRecords filename."
)

flags.DEFINE_string(
    "eval_filename", "eval.tfrecords@*", "Test examples TFRecords filename."
)

flags.DEFINE_string("model_dir", "", "Path to the model folder.")

flags.DEFINE_integer("eval_batch_size", 8, "Batch size for eval.")

flags.DEFINE_integer("steps_between_saves", 1000, "How many steps between saves.")

flags.DEFINE_integer("max_eval_steps", None, "Number of evaluation steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use a TPU for training.")

flags.DEFINE_string("primary", "", "The primary machine to use for TPU training.")

flags.DEFINE_integer(
    "num_tpu_shards", 1, "The number of shards to use during TPU training."
)

KEEP_CHECKPOINTS_MAX = 5


class ValidationHook(tf.train.SessionRunHook):
    """

    """

    def __init__(
        self,
        model_fn,
        params,
        input_fn,
        model_dir,
        experiment,
        every_n_secs=None,
        every_n_steps=None,
    ):
        self._iter_count = 0
        self._estimator = tf.estimator.Estimator(
            model_fn=model_fn, params=params, model_dir=model_dir
        )
        self._input_fn = input_fn
        self._experiment = experiment
        self._timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
        self._should_trigger = False

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            eval_results = self._estimator.evaluate(
                input_fn=self._input_fn, steps=FLAGS.max_eval_steps,
            )
            self._experiment.checkpoint(
                metrics={
                    "eval_loss": eval_results["loss"],
                    "accuracy": eval_results["sequence_correct"],
                },
                step=self._iter_count,
                primary_metric=("eval_loss", "minimize"),
            )
            tf.logging.info("Eval results: %s", eval_results)
            self._timer.update_last_triggered_step(self._iter_count)
        self._iter_count += 1


def global_seed(seed):
    tf.random.set_random_seed(seed)


def get_ckpt_number(ckpt):
    pattern = re.compile("model.ckpt-[0-9]+")
    pattern_match = pattern.search(ckpt)
    assert pattern_match is not None, f"'{ckpt}' should match '{pattern}'"
    return int(pattern_match.group().replace("model.ckpt-", ""))


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        gfile.Copy(source + ext, target + ext)


def evaluate(estimator, eval_input_fn, checkpoint):
    """
    Call estimator.evaluate with a given checkpoint
    """
    try:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        if FLAGS.max_eval_steps:
            tf.logging.info("  Eval steps = %d", FLAGS.max_eval_steps)

        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=FLAGS.max_eval_steps,
            checkpoint_path=checkpoint,
            name=FLAGS.eval_filename.split(".")[0],
        )
        tf.logging.info("Eval results: %s", eval_results)

        return eval_results
    except tf.errors.NotFoundError:
        tf.logging.info("Checkpoint %s no longer exists, skipping.", checkpoint)


def main(unused_argv):
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
        config, FLAGS.output_vocab, clean_output_vocab_path=""
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

    # region training
    if FLAGS.do_train:
        # for keepsake CLI
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

        eval_input_fn = input_pipeline.create_eval_input_fn(
            config, FLAGS.tf_examples_dir, [FLAGS.eval_filename], FLAGS.eval_batch_size
        )

        validation_hook = ValidationHook(
            model_fn=model_fn,
            params={},
            input_fn=eval_input_fn,
            model_dir=FLAGS.model_dir,
            every_n_steps=FLAGS.steps_between_saves,
            experiment=experiment,
        )

        estimator.train(
            input_fn=train_input_fn,
            max_steps=config.training_options.training_steps,
            hooks=[validation_hook],
        )
    # endregion

    # region eval
    if FLAGS.do_eval:
        max_acc = 0.0

        eval_input_fn = input_pipeline.create_eval_input_fn(
            config, FLAGS.tf_examples_dir, [FLAGS.eval_filename]
        )

        num_train_steps = int(config.training_options.training_steps)

        for ckpt in tf.estimator.training.checkpoints_iterator(FLAGS.model_dir):
            acc = evaluate(estimator, eval_input_fn, ckpt)

            if acc > max_acc:
                copy_checkpoint(
                    ckpt,
                    os.path.join(
                        FLAGS.model_dir,
                        str(get_ckpt_number(ckpt))
                        + "model_max_"
                        + FLAGS.eval_filename.split(".")[0]
                        + ".ckpt",
                    ),
                )

            if get_ckpt_number(ckpt) == num_train_steps:
                break
    # endregion


if __name__ == "__main__":
    app.run(main)
