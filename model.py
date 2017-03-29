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
Replicate "Very Deep Convolutional Networks for Natural Language Processing" by Alexis Conneau,
Holger Schwenk, Yann Le Cun, Loic Barraeau, 2016
https://arxiv.org/pdf/1606.01781.pdf

New NLP architecture:
1. Operate at lowest atomic representation of text (characters)
2. Use deep-stack of local operations to learn high-level hierarchical representation
"""
import tensorflow as tf
import numpy as np
import config
# import ops
from config import FEATURE_LEN
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
FLAGS = tf.app.flags.FLAGS

VDCNN_VARIABLES = 'vdcnn_variables'
UPDATE_OPS_COLLECTION = 'vdcnn_update_ops'  # must be grouped with training op


class VDCNN(object):
    """
    very deep CNN for text classification.
    """
    def __init__(
      self, sequence_length,
            num_classes,
            vocab_size,
            embedding_size,
            is_training,
            l2_reg_lambda=0.0):

        s = FEATURE_LEN
        f0 = 1
        embedding_size = 16
        temp_kernel = (3, embedding_size)
        kernel = (3, 1)
        stride = (2, 1)
        # kmax = 8
        num_filters1 = 64
        num_filters2 = 128
        num_filters3 = 256
        num_filters4 = 512
        activation = tf.nn.relu
        fc1_hidden_size = 1024
        fc2_hidden_size = 512
        num_output = num_classes


        self.is_training = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, s], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # self.loss = .0
        self._extra_train_ops = []
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")#TODO char load embedding
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            embedded_f0_channel = tf.reshape(embedded_chars_expanded,[-1,f0,s,embedding_size])

    # with tf.device('/gpu:0'):
        # Temp Conv (in: batch, fo, FEATURE_LEN, embd_size)
        conv0 = self._conv(x=embedded_f0_channel, kernel=temp_kernel,
                      stride=stride, filters_out=num_filters1,name='conv0')
        act0 = activation(features=conv0, name='relu')

        # CONVOLUTION_BLOCK (1 of 4) -> 64 FILTERS
        with tf.variable_scope('block1'):
            conv11 = self._conv(x=act0, kernel=kernel, stride=stride,filters_out=num_filters1,name='conv11')
            norm11 = self._batch_norm(conv11, self.is_training, name='norm11')
            act11 = activation(features=norm11, name='relu')
            conv12 = self._conv(x=act11, kernel=kernel, stride=stride,filters_out=num_filters1,name='conv12')
            norm12 = self._batch_norm(conv12, self.is_training, name='norm12')
            act12 = activation(features=norm12, name='relu')

            # conv21 = ops._conv(x=act12, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv21')
            # norm21 = ops._bn(conv21, self.is_training, name='norm21')
            # act21 = activation(features=norm21, name='relu')
            # conv22 = ops._conv(x=act21, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv22')
            # norm22 = ops._bn(conv22, self.is_training, name='norm22')
            # act22 = activation(features=norm22, name='relu')

            # conv31 = ops._conv(x=act22, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv31')
            # norm31 = ops._bn(conv31, self.is_training, name='norm31')
            # act31 = activation(features=norm31, name='relu')
            # conv32 = ops._conv(x=act31, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv32')
            # norm32 = ops._bn(conv32, self.is_training, name='norm32')
            # act32 = activation(features=norm32, name='relu')
            #
            # conv41 = ops._conv(x=act32, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv41')
            # norm41 = ops._bn(conv41, self.is_training, name='norm41')
            # act41 = activation(features=norm41, name='relu')
            # conv42 = ops._conv(x=act41, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv42')
            # norm42 = ops._bn(conv42, self.is_training, name='norm42')
            # act42 = activation(features=norm42, name='relu')
            #
            # conv51 = ops._conv(x=act42, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv51')
            # norm51 = ops._bn(conv51, self.is_training, name='norm51')
            # act51 = activation(features=norm51, name='relu')
            # conv52 = ops._conv(x=act51, kernel=kernel, stride=stride, filters_out=num_filters1, name='conv52')
            # norm52 = ops._bn(conv52, self.is_training, name='norm52')
            # act52 = activation(features=norm52, name='relu')



        # CONVOLUTION_BLOCK (2 of 4) -> 128 FILTERS
        with tf.variable_scope('block2'):
            conv61 = self._conv(x=act12, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv61')
            norm61 = self._batch_norm(conv61, self.is_training, name='norm61')
            act61 = activation(features=norm61, name='relu')
            conv62 = self._conv(x=act61, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv62')
            norm62 = self._batch_norm(conv62, self.is_training, name='norm62')
            act62 = activation(features=norm62, name='relu')

            # conv71 = ops._conv(x=act62, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv71')
            # norm71 = ops._bn(conv71, self.is_training, name='norm71')
            # act71 = activation(features=norm71, name='relu')
            # conv72 = ops._conv(x=act71, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv72')
            # norm72 = ops._bn(conv72, self.is_training, name='norm72')
            # act72 = activation(features=norm72, name='relu')

            # conv81 = ops._conv(x=act72, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv81')
            # norm81 = ops._bn(conv81, self.is_training, name='norm81')
            # act81 = activation(features=norm81, name='relu')
            # conv82 = ops._conv(x=act81, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv82')
            # norm82 = ops._bn(conv82, self.is_training, name='norm82')
            # act82 = activation(features=norm82, name='relu')
            #
            # conv91 = ops._conv(x=act82, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv91')
            # norm91 = ops._bn(conv91, self.is_training, name='norm91')
            # act91 = activation(features=norm91, name='relu')
            # conv92 = ops._conv(x=act91, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv92')
            # norm92 = ops._bn(conv92, self.is_training, name='norm92')
            # act92 = activation(features=norm92, name='relu')
            #
            # conv101 = ops._conv(x=act92, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv101')
            # norm101 = ops._bn(conv101, self.is_training, name='norm101')
            # act101 = activation(features=norm101, name='relu')
            # conv102 = ops._conv(x=act101, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv102')
            # norm102 = ops._bn(conv102, self.is_training, name='norm102')
            # act102 = activation(features=norm102, name='relu')
    # with tf.device("/gpu:1"):
        # CONVOLUTION_BLOCK (3 of 4) -> 256 FILTERS
        with tf.variable_scope('block3'):
            conv111 = self._conv(x=act62, kernel=kernel, stride=stride, filters_out=num_filters3, name='conv111')
            norm111 = self._batch_norm(conv111, self.is_training, name='norm111')
            act111 = activation(features=norm111, name='relu')
            conv112 = self._conv(x=act111, kernel=kernel, stride=stride, filters_out=num_filters3,name='conv112')
            norm112 = self._batch_norm(conv112, self.is_training, name='norm112')
            act112 = activation(features=norm112, name='relu')

        # conv121 = self._conv(x=act112, kernel=kernel, stride=stride, filters_out=num_filters3, name='conv121')
        # norm121 = self._bn(conv121, self.is_training, name='norm121')
        # act121 = activation(features=norm121, name='relu')
        # conv122 = self._conv(x=act121, kernel=kernel, stride=stride, filters_out=num_filters3, name='conv122')
        # norm122 = self._bn(conv122, self.is_training, name='norm122')
        # act122 = activation(features=norm122, name='relu')


        # CONVOLUTION_BLOCK (4 of 4) -> 512 FILTERS
        with tf.variable_scope('block4'):
            conv131 = self._conv(x=act112, kernel=kernel, stride=stride, filters_out=num_filters4,name='conv131')
            norm131 = self._batch_norm(conv131, self.is_training, name='norm131')
            act131 = activation(features=norm131, name='relu')
            conv132 = self._conv(x=act131, kernel=kernel, stride=stride, filters_out=num_filters4,name='conv132')
            norm132 = self._batch_norm(conv132, self.is_training, name='norm132')
            act132 = activation(features=norm132, name='relu')

            # conv141 = ops._conv(x=act132, kernel=kernel, stride=stride, filters_out=num_filters4, name='conv141')
            # norm141 = ops._bn(conv141, self.is_training, name='norm141')
            # act141 = activation(features=norm141, name='relu')
            # conv142 = ops._conv(x=act141, kernel=kernel, stride=stride, filters_out=num_filters4, name='conv142')
            # norm142 = ops._bn(conv142, self.is_training, name='norm142')
            # act142 = activation(features=norm142, name='relu')

        # K-max pooling (k=8) ? vs max pooling , according to paper,
        # Max-pooling performs better than other pool-
        # ing types. In terms of pooling, we can also see
        # that max-pooling performs best overall, very close
        # to convolutions with stride 2, but both are signifi-
        # cantly superior to k-max pooling.
        max_pool = self._max_pool(act132)
        flatten = tf.reshape(max_pool, [-1,  s * num_filters4])#TODO figure out the correct multiplier

        # Fully connected layers (fc)
        # fc1
        fc1 = self._fc(flatten, fc1_hidden_size, 'fc1')
        act_fc1 = activation(fc1)

        # fc2
        fc2 = self._fc(act_fc1, fc2_hidden_size, 'fc2')
        act_fc2 = activation(fc2)

        # CalculateMean cross-entropy loss
        logits = self._fc(act_fc2, num_output, 'softmax')
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        predictions = tf.argmax(logits, 1, name="prediction")
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def build_train_op(self,lr,global_step):
        """Build training specific ops for the graph."""
        tf.summary.scalar('learning_rate', lr)

        trainable_variables = tf.trainable_variables()
        # grads = tf.gradients(self.loss, trainable_variables)

        # if self.hps.optimizer == 'sgd':
        #     optimizer = tf.train.GradientDescentOptimizer(lr)
        # elif self.hps.optimizer == 'mom':
        #     optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        #
        # apply_op = optimizer.apply_gradients(
        #     zip(grads, trainable_variables),
        #     global_step=self.global_step, name='train_step')

        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        optimizer_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        train_ops = [optimizer_op] + self._extra_train_ops
        return tf.group(*train_ops)

    def _fc(self,x, units_out, name):

        num_units_in = x.get_shape()[1]
        num_units_out = units_out

        weights_initializer = tf.truncated_normal_initializer(
            stddev=config.FC_WEIGHT_STDDEV)

        weights = self._get_variable(name + str(units_out) + 'weights',
                                shape=[num_units_in, num_units_out],
                                initializer=weights_initializer,
                                weight_decay=config.FC_WEIGHT_STDDEV)
        biases = self._get_variable(name + str(units_out) + 'biases',
                               shape=[num_units_out],
                               initializer=tf.zeros_initializer())
        x = tf.nn.xw_plus_b(x, weights, biases)

        return x

    def _get_variable(self,
                      name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        """A little wrapper around tf.get_variable to
         do weight decay and add to resnet collection"""

        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, VDCNN_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

    def _conv(self,x, kernel, stride, filters_out, name):
        """
        :param x:
        :param temp_kernel:
        :param stride:
        :param filters_out:
        :return:
        """
        filters_in = x.get_shape()[-1]
        shape = [kernel[0], kernel[1], filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=config.CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights' + "_" + name,
                                shape=shape,
                                dtype='float',
                                initializer=initializer,
                                weight_decay=config.CONV_WEIGHT_DECAY)

        return tf.nn.conv2d(x, weights, [1, stride[0], stride[1], 1], padding='SAME')

    def _max_pool(self,x, ksize=1, stride=1):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='VALID')

    def _batch_norm(self,x, is_training, name):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable('beta' + "_" + name,
                             params_shape,
                             initializer=tf.zeros_initializer())
        gamma = self._get_variable('gamma' + "_" + name,
                              params_shape,
                              initializer=tf.ones_initializer())

        moving_mean = self._get_variable('moving_mean' + "_" + name,
                                    params_shape,
                                    initializer=tf.zeros_initializer(),
                                    trainable=False)
        moving_variance = self._get_variable('moving_variance' + "_" + name,
                                        params_shape,
                                        initializer=tf.ones_initializer(),
                                        trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, config.BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, config.BN_DECAY)
        self._extra_train_ops.append(update_moving_mean)
        self._extra_train_ops.append(update_moving_variance)

        mean, variance = control_flow_ops.cond(
            is_training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, config.BN_EPSILON)

        return x

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


