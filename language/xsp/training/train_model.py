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
from absl import app, flags
import tensorflow.compat.v1.gfile as gfile

import os
import re
from pathlib import Path

import language.xsp.model.input_pipeline as input_pipeline
import language.xsp.model.model_builder as model_builder
import language.xsp.model.model_config as model_config
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_bool("do_train", False, "Whether to do training.")

flags.DEFINE_bool("do_eval", False, "Whether to do evaluation continuously.")

flags.DEFINE_string("tf_examples_dir", "",
                    "Path to the tensorflow examples folder.")

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

flags.DEFINE_integer("steps_between_saves", 1000,
                     "How many steps between saves.")

flags.DEFINE_integer("max_eval_steps", None, "Number of evaluation steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use a TPU for training.")

flags.DEFINE_string(
    "primary", "", "The primary machine to use for TPU training.")

flags.DEFINE_integer(
    "num_tpu_shards", 1, "The number of shards to use during TPU training."
)

KEEP_CHECKPOINTS_MAX = 5


def get_ckpt_number(ckpt):
    pattern = re.compile("model.ckpt-[0-9]+")
    pattern_match = pattern.search(ckpt)
    assert pattern_match is not None
    return int(pattern_match.group().replace("model.ckpt-", ""))


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        gfile.Copy(source + ext, target + ext)


def evaluate(estimator, eval_input_fn, checkpoint):
    """Call estimator.evaluatior with a given checkpoint."""
    try:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        if FLAGS.max_eval_steps:
            tf.logging.info("  eval step= %d", FLAGS.max_eval_steps)
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=FLAGS.max_eval_steps,
            checkpoint_path=checkpoint,
            name=FLAGS.eval_filename.split(".")[0],
        )
        tf.logging.info("Eval results: %s", eval_results)

        return eval_results["sequence_correct"]
    except tf.errors.NotFoundError:
        tf.logging.info(
            "Checkpoint %s no longer exists, skipping.", checkpoint)


class OomReportingHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=[],  # no extra fetches
                                       options=tf.RunOptions(
            report_tensor_allocations_upon_oom=True))


def main(unused_argv):
    tf.logging.info("Saving model saves and results to " + FLAGS.model_dir)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train`, `do_eval` must be True.")

    config = model_config.load_config(FLAGS.config)

    export_dir = str(Path(FLAGS.model_dir) / "ckpt")

    if FLAGS.do_train:
        tf.logging.info(
            "Training with train filenames: " + str(FLAGS.training_filename)
        )

    training_options = config.training_options
    use_tpu = FLAGS.use_tpu
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=1,
        save_checkpoints_steps=FLAGS.steps_between_saves,
        keep_checkpoint_max=KEEP_CHECKPOINTS_MAX,
    )

    # Set up estimator, in training allows noisy examples so do not use
    # clean output vocab
    model_fn = model_builder.build_model_fn(
        config, FLAGS.output_vocab, clean_output_vocab_path=""
    )

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    if FLAGS.do_train:
        train_input_fn = input_pipeline.create_training_input_fn(
            config,
            FLAGS.tf_examples_dir,
            [name for name in FLAGS.training_filename if name],
            False,  # use_tpu
        )

        with tf.Session() as sess:
            features, labels = train_input_fn({})
            model_fn_results = model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

            for step in range(config.training_options.training_steps):
                _, loss = sess.run(
                    [model_fn_results.train_op, model_fn_results.loss])

                if step % 1000 == 0:
                    print("step:", step, "loss:", loss)
                    save_path = saver.save(sess, export_dir )

            save_path = saver.save(sess, export_dir)
            print(step, save_path)

    if FLAGS.do_eval:
        max_acc = 0.0

        eval_input_fn = input_pipeline.create_eval_input_fn(
            config, FLAGS.tf_examples_dir, [
                FLAGS.eval_filename], False,  # use_tpu
        )

        # When FLAGS.init_checkpoint = None, the latest checkpoint will be evaluated
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


if __name__ == "__main__":
    app.run(main)
