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
import numpy as np
import os
import time
import datetime
import sys

import util
from model import VDCNN
import config

import tensorflow as tf
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .4, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_float("test_sample_percentage", .0, "Percentage of the training data to use for test")
tf.flags.DEFINE_string("train_data_file", "./data/rt-polaritydata/rt_data_all.txt", "train Data source")
tf.flags.DEFINE_string("test_data_file", "./data/sogou_news_csv/sogou_data_test.txt", "test Data source")

#rt-polaritydata/rt_data_all.txt
#sogou_news_csv/sogou_data_train_dev.txt
# Model Hyperparameters
tf.flags.DEFINE_integer("feature_len", config.FEATURE_LEN, "maximum length of the sentence at char level")
tf.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_float("lr", 1e-4, "learning rate")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("TRAIN_DIR", "train_dir", "training directory to store training results")
tf.flags.DEFINE_boolean("resume", False, "whether resume training from the previous checkpoints")
tf.flags.DEFINE_string("CHECKPOINT_DIR", "/home/wenchen/projects/VDCNN/train_dir/1490807032/checkpoints",
                       "checkpoint dir for model to resume training")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# tf.flags.DEFINE_string("mode", 'test', "train or test")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# log current train parameters
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation train/test
# ==================================================

# Load data
print("Loading data...")

# if FLAGS.mode == 'train':

x_text, y, index2label = util.load_data_and_labels_fasttext(FLAGS.train_data_file)
#TODO index2label used for predict

# Build vocabulary and transform the corpus

vocabulary = learn.preprocessing.CategoricalVocabulary()
for token in config.ALPHABET:
    vocabulary.add(token)
vocabulary.freeze()

max_document_length = config.FEATURE_LEN
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,vocabulary=vocabulary, tokenizer_fn=list)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/dev set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = \
    x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = \
    y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
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
        vdcnn = VDCNN(
            feature_len=FLAGS.feature_len,
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            is_training=is_training,
            depth=9)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_ops = vdcnn.build_train_op(FLAGS.lr,global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.TRAIN_DIR, timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", vdcnn.loss)
        acc_summary = tf.summary.scalar("accuracy", vdcnn.accuracy)

        # Train Summaries
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # resume or Initialize all variables to train from scratch
        if FLAGS.resume:
            latest = str(util.latest_checkpoint(FLAGS.CHECKPOINT_DIR))
            if not latest:
                print("No checkpoint to continue from in", latest)
                sys.exit(1)
            print("resume training", latest)
            saver.restore(sess, latest)
        else:
            sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              vdcnn.input_x: x_batch,
              vdcnn.input_y: y_batch,
              is_training: True
            }

            _, step, loss, accuracy = sess.run(
                [train_ops, global_step, vdcnn.loss, vdcnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            losses = []
            accuracies = []
            start_index = 0
            end_index = start_index + FLAGS.batch_size
            for i in xrange(len(y_batch)/FLAGS.batch_size + 1):

                feed_dict = {
                  vdcnn.input_x: x_batch[start_index:end_index],
                  vdcnn.input_y: y_batch[start_index:end_index],
                  is_training: False
                }

                loss, accuracy = sess.run(
                    [vdcnn.loss, vdcnn.accuracy],
                    feed_dict)

                start_index = end_index
                end_index += FLAGS.batch_size
                losses.append(loss)
                accuracies.append(accuracy)

            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g}".format(time_str,  np.mean(losses), np.mean(accuracies)))

        def do_test(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                vdcnn.input_x: x_batch,
                vdcnn.input_y: y_batch,
                is_training: False
            }

            step, loss, accuracy = sess.run(
                [global_step, vdcnn.loss, vdcnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

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
                dev_step(x_dev, y_dev)
                print("----------------------------------------")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

