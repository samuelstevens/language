"""
A simple federated averaging algorithm.
"""
from pathlib import Path

import tensorflow.compat.v1 as tf

import language.xsp.model.input_pipeline as input_pipeline
import language.xsp.model.model_builder as model_builder
import language.xsp.training.train_utils as train_utils


def train_federated(config, flags):
    """
    Trains the model with fed avg.
    """

    client_scope = "client-scope"
    server_scope = "server-scope"
    sum_scope = "sum-scope"

    export_dir = str(Path(flags.model_dir) / "ckpt")

    train_input_fn = input_pipeline.create_training_input_fn(
        config,
        flags.tf_examples_dir,
        [name for name in flags.training_filename if name],
        federated=True,
    )

    features, labels, client_iterators, train_placeholder = train_input_fn()

    # Training allows noisy examples so do not use clean output vocab
    model_fn = model_builder.build_model_fn(
        config, flags.output_vocab, clean_output_vocab_path=""
    )

    with tf.variable_scope(client_scope):
        client_model = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)

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
                    _, loss = sess.run(
                        [client_model.train_op, client_model.loss],
                        feed_dict={train_placeholder: client_handle},
                    )
                    print(
                        f"Outer step: {step}; Client: {k}; Client step: {client_step}; Loss: {loss}"
                    )

                # add wk* to sum(wk*)
                sess.run(train_utils.add_params(client_scope, sum_scope))

            # wt+1 = mean(sum(wk*))
            sess.run(
                train_utils.mean_and_assign_params(
                    sum_scope, server_scope, len(client_handles)
                )
            )

            if step % flags.steps_between_saves == 0:
                print("step:", step, "loss:", loss)
                print("Saving...")
                save_path = saver.save(sess, export_dir)
                print(f"Saved to {save_path}")

        # endregion
