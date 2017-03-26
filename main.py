# Copyright 2017 The Wenchen Li. All Rights Reserved.
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
# ==============================================================================
"""
run the training/dev/testing
"""
import tensorflow as tf
import time
import numpy as np
import os
import time
import datetime
import util
from model import TextCNN,VDCNN
from tensorflow.contrib import learn
import config

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = util.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))

max_document_length = config.FEATURE_LEN
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,vocabulary=config.ALPHABET,tokenizer_fn=list)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=list)#TODO  vocabularyBuilder contains full char defined in config ALPHABET
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    is_training = tf.placeholder('bool', [], name='is_training')

    with sess.as_default():
        cnn = VDCNN(
            sequence_length=x_train.shape[1], #x_train.shape[1]
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            is_training=is_training)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              is_training: True
            }

            #   cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            # }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              is_training:True
            }
            #   cnn.dropout_keep_prob: 1.0
            # }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = util.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


# def save_check_point(mod_arg, mod_aux, pre, epoch):
#     """
#     Save model each epoch, load as:
#
#     sym, arg_params, aux_params = \
#         mx.model.load_checkpoint(model_prefix, n_epoch_load)
#
#     # assign parameters
#     mod.set_params(arg_params, aux_params)
#
#     OR
#
#     mod.fit(..., arg_params=arg_params, auxdata_helpers
# data_helpers_params=aux_params,
#             begin_epoch=n_epoch_load)
#     """
#
#     save_dict = {('arg:%s' % k): v for k, v in mod_arg.items()}
#     save_dict.update({('aux:%s' % k): v for k, v in mod_aux.items()})
#     param_name = '%s-%04d.pk' % (pre, epoch)
#     pickle.dump(save_dict, open(param_name, "wb"))
#     print('Saved checkpoint to \"%s\"' % param_name)
#
#
# def load_check_point(file_name):
#     # Load file
#     print(file_name)
#     save_dict = pickle.load(open(file_name, "rb"))
#     # Extract data from save
#     arg_params = {}
#     aux_params = {}
#     for k, v in save_dict.items():
#         tp, name = k.split(':', 1)
#         if tp == 'arg':
#             arg_params[name] = v
#         if tp == 'aux':
#             aux_params[name] = v
#
#     # Recreate model
#     cnn = create_vdcnn()
#     mod = mx.mod.Module(cnn, context=ctx)
#
#     # Bind shape
#     mod.bind(data_shapes=[('data', DATA_SHAPE)],
#              label_shapes=[('softmax_label', (BATCH_SIZE,))])
#
#     # assign parameters from save
#     mod.set_params(arg_params, aux_params)
#     print('Model loaded from disk')
#
#     return mod
#
#
# def train_model(train_fname):
#     # Create mx.mod.Module()
#     cnn = create_vdcnn()
#     mod = mx.mod.Module(cnn, context=ctx)
#
#     # Bind shape
#     mod.bind(data_shapes=[('data', DATA_SHAPE)],
#              label_shapes=[('softmax_label', (BATCH_SIZE,))])
#
#     # Initialise parameters and optimiser
#     mod.init_params(mx.init.Normal(sigma=SD))
#     mod.init_optimizer(optimizer='sgd',
#                        optimizer_params={
#                            "learning_rate": 0.01,
#                            "momentum": 0.9,
#                            "wd": 0.00001,
#                            "rescale_grad": 1.0 / BATCH_SIZE
#                        })
#
#     # Load Data
#     X_train, y_train = load_file('amazon_review_polarity_train.csv')
#
#     # Train
#     print("Alphabet %d characters: " % len(ALPHABET), ALPHABET)
#     print("started training")
#     tic = time.time()
#
#     # Evaluation metric:
#     metric = mx.metric.Accuracy()
#
#     # Train EPOCHS
#     for epoch in range(EPOCHS):
#         t = 0
#         metric.reset()
#         tic_in = time.time()
#         for batch in load_data_frame(X_data=X_train,
#                                      y_data=y_train,
#                                      batch_size=BATCH_SIZE,
#                                      shuffle=True):
#             # Push data forwards and update metric
#             mod.forward_backward(batch)
#             mod.update()
#             mod.update_metric(metric, batch.label)
#
#             # For training + testing
#             # mod.forward(batch, is_train=True)
#             # mod.update_metric(metric, batch.label)
#             # Get weights and update
#             # For training only
#             # mod.backward()
#             # mod.update()
#             # Log every 50 batches = 128*50 = 6400
#             t += 1
#             if t % 50 == 0:
#                 train_t = time.time() - tic_in
#                 metric_m, metric_v = metric.get()
#                 print("epoch: %d iter: %d metric(%s): %.4f dur: %.0f" % (epoch, t, metric_m, metric_v, train_t))
#
#         # Checkpoint
#         arg_params, aux_params = mod.get_params()
#         save_check_point(mod_arg=arg_params,
#                          mod_aux=aux_params,
#                          pre=train_fname,
#                          epoch=epoch)
#         print("Finished epoch %d" % epoch)
#
#     print("Done. Finished in %.0f seconds" % (time.time() - tic))
#
#
# def test_model(test_fname):
#     # Load saved model:
#     mod = load_check_point(test_fname)
#     # assert mod.binded and mod.params_initialized
#
#     # Load data
#     X_test, y_test = load_file('amazon_review_polarity_test.csv')
#
#     # Score accuracy
#     metric = mx.metric.Accuracy()
#
#     # Test batches
#     for batch in load_data_frame(X_data=X_test,
#                                  y_data=y_test,
#                                  batch_size=len(y_test)):
#         mod.forward(batch, is_train=False)
#         mod.update_metric(metric, batch.label)
#
#         metric_m, metric_v = metric.get()
#         print("TEST(%s): %.4f" % (metric_m, metric_v))