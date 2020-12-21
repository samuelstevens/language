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
import language.xsp.training.train_utils as train_utils
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

    client_scope = 'client-scope'
    server_scope = 'server-scope'
    sum_scope = 'sum-scope'

    if FLAGS.do_train:
        tf.logging.info(
            "Training with train filenames: " + str(FLAGS.training_filename)
        )

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
        )

        features, labels, client_iterators, train_placeholder = train_input_fn()

        with tf.variable_scope(client_scope):
            client_model = model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope(server_scope):
            server_model = model_fn(features, labels, tf.estimator.ModeKeys.EVAL)

        with tf.variable_scope(sum_scope):
            model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
      
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            client_handles = []
            for iterator in client_iterators:
                handle = sess.run(iterator.string_handle())
                client_handles.append(handle)

            saver = tf.train.Saver()

            # region main training loop

            for step in range(config.training_options.training_steps):
                sess.run(train_utils.zero(sum_scope))

                for k, client_handle in enumerate(client_handles):
                    # put wt on ck
                    sess.run(train_utils.copy_params(server_scope, client_scope))

                    # train on dk for E iterations to produce wk*
                    for client_step in range(config.training_options.client_steps):
                        _, loss = sess.run([client_model.train_op, client_model.loss], feed_dict={train_placeholder: client_handle})
                        print(f'Outer step: {step}; Client: {k}; Client step: {client_step}; Loss: {loss}')

                    # add wk* to sum(wk*)
                    sess.run(train_utils.add_params(client_scope, sum_scope))

                # wt+1 = mean(sum(wk*))
                sess.run(train_utils.mean_and_assign_params(sum_scope, server_scope, len(client_handles)))

                if step % FLAGS.steps_between_saves == 0:
                    print("step:", step, "loss:", loss)
                    print("Saving...")
                    save_path = saver.save(sess, export_dir)

            # endregion

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
