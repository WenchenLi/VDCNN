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
import ops
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

RESNET_VARIABLES = 'vdcnn_variables'
UPDATE_OPS_COLLECTION = 'vdcnn_update_ops'  # must be grouped with training op
FLAGS = tf.app.flags.FLAGS

class Config:
    def __init__(self):
        root = self.Scope('')
        for k, v in FLAGS.__dict__['__flags'].iteritems():
            root[k] = v
        self.stack = [ root ]

    def iteritems(self):
        return self.to_dict().iteritems()

    def to_dict(self):
        self._pop_stale()
        out = {}
        # Work backwards from the flags to top fo the stack
        # overwriting keys that were found earlier.
        for i in range(len(self.stack)):
            cs = self.stack[-i]
            for name in cs:
                out[name] = cs[name]
        return out

    def _pop_stale(self):
        var_scope_name = tf.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def set_default(self, name, value):
        if not name in self:
            self[name] = value

    def __contains__(self, name):
        self._pop_stale()
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return True
        return False

    def __setitem__(self, name, value):
        self._pop_stale()
        top = self.stack[0]
        var_scope_name = tf.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)


# def get_resnet_training_model(is_training,var_reuse=None):
#
#     x, conv_layer, c = resenet_conv_layer(is_training,variable_reuse=var_reuse)
#
#     # S
#     with tf.variable_scope('S',reuse=c['variable_reuse']):
#         S = []
#         for i in xrange(config.MAX_LEN_CHARS ):
#             with tf.variable_scope(str(i) + '_mlp',reuse=c['variable_reuse']):
#                 bn_ = bn(activation(fc(conv_layer, c, 2048)),c)
#                 fc_ = fc(bn_, c, "S")
#                 S.append(tf.nn.softmax(fc_))
#         S_logits = ops.concatenate(S, "merged_S")
#         S_logits = tf.reshape(S_logits,(-1,config.MAX_LEN_CHARS,config.NUM_CLASSES))
#         y = S_logits
#
#     # L
#     with tf.variable_scope('L_mlp',reuse=c['variable_reuse']):
#         out_L = bn(activation(fc(conv_layer, c, 2048)),c)
#         L_logits = fc(out_L, c, "L")
#         l = L_logits
#
#     return (x, y, l)
#
# def resenet_conv_layer(is_training,
#                           L_num_classes=7,
#                           S_num_classes=63,
#                           num_blocks=None,  # defaults to 50-layer network
#                           use_bias=False,  # defaults to using batch norm
#                           bottleneck=True,
#                         variable_reuse=None):
#     if num_blocks is None:
#         num_blocks = [3, 4, 6, 3]
#
#     c = Config()
#     c['bottleneck'] = bottleneck
#     c['is_training'] = tf.convert_to_tensor(is_training,
#                                             dtype='bool',
#                                             name='is_training')
#     c['ksize'] = 3
#     c['stride'] = 1
#     c['use_bias'] = use_bias
#     c['L_fc_units_out'] = L_num_classes
#     c['S_fc_units_out'] = S_num_classes
#     c['num_blocks'] = num_blocks
#     c['stack_stride'] = 2
#     c['variable_reuse'] = variable_reuse
#
#     x = tf.placeholder(tf.float32, [None, 200, 200, config.NUM_CHANNELS])
#
#     with tf.variable_scope('scale1',reuse=c['variable_reuse']):
#         c['conv_filters_out'] = 64
#         c['ksize'] = 7
#         c['stride'] = 2
#         out = conv(x, c)
#         out = bn(out, c)
#         out = activation(out)
#         out = _max_pool(out, ksize=3, stride=2)
#
#
#     with tf.variable_scope('scale2',reuse=c['variable_reuse']):
#         c['num_blocks'] = num_blocks[0]
#         c['stack_stride'] = 1
#         c['block_filters_internal'] = 64
#         assert c['ksize'] == 3
#         out = stack(out, c)
#
#     with tf.variable_scope('scale3',reuse=c['variable_reuse']):
#         c['num_blocks'] = num_blocks[1]
#         c['block_filters_internal'] = 128
#         assert c['ksize'] == 3
#         out = stack(out, c)
#
#     with tf.variable_scope('scale4',reuse=c['variable_reuse']):
#         c['num_blocks'] = num_blocks[2]
#         c['block_filters_internal'] = 256
#         assert c['ksize'] == 3
#         out = stack(out, c)
#
#     with tf.variable_scope('scale5',reuse=c['variable_reuse']):
#         c['num_blocks'] = num_blocks[3]
#         c['block_filters_internal'] = 512
#         assert c['ksize'] == 3
#         out = stack(out, c)
#
#     with tf.variable_scope('post-net',reuse=c['variable_reuse']):
#         avg_pool = tf.reduce_mean(out, reduction_indices=[1, 2], name="avg_pool")
#
#     return x, avg_pool, c
#
#
# def stack(x, c):
#     for n in range(c['num_blocks']):
#         s = c['stack_stride'] if n == 0 else 1
#         c['block_stride'] = s
#         with tf.variable_scope('block%d' % (n + 1),reuse=c['variable_reuse']):
#             x = block(x, c)
#     return x
#
#
# def block(x, c):
#     filters_in = x.get_shape()[-1]
#
#     # Note: filters_out isn't how many filters are outputed.
#     # That is the case when bottleneck=False but when bottleneck is
#     # True, filters_internal*4 filters are outputted. filters_internal is how many filters
#     # the 3x3 convs output internally.
#     m = 4 if c['bottleneck'] else 1
#     filters_out = m * c['block_filters_internal']
#
#     shortcut = x  # branch 1
#
#     c['conv_filters_out'] = c['block_filters_internal']
#
#     if c['bottleneck']:
#         with tf.variable_scope('a',reuse=c['variable_reuse']):
#             c['ksize'] = 1
#             c['stride'] = c['block_stride']
#             x = conv(x, c)
#             x = bn(x, c)
#             x = activation(x)
#
#         with tf.variable_scope('b',reuse=c['variable_reuse']):
#             x = conv(x, c)
#             x = bn(x, c)
#             x = activation(x)
#
#         with tf.variable_scope('c',reuse=c['variable_reuse']):
#             c['conv_filters_out'] = filters_out
#             c['ksize'] = 1
#             assert c['stride'] == 1
#             x = conv(x, c)
#             x = bn(x, c)
#     else:
#         with tf.variable_scope('A',reuse=c['variable_reuse']):
#             c['stride'] = c['block_stride']
#             assert c['ksize'] == 3
#             x = conv(x, c)
#             x = bn(x, c)
#             x = activation(x)
#
#         with tf.variable_scope('B',reuse=c['variable_reuse']):
#             c['conv_filters_out'] = filters_out
#             assert c['ksize'] == 3
#             assert c['stride'] == 1
#             x = conv(x, c)
#             x = bn(x, c)
#
#     with tf.variable_scope('shortcut',reuse=c['variable_reuse']):
#         if filters_out != filters_in or c['block_stride'] != 1:
#             c['ksize'] = 1
#             c['stride'] = c['block_stride']
#             c['conv_filters_out'] = filters_out
#             shortcut = conv(shortcut, c)
#             shortcut = bn(shortcut, c)
#
#     return activation(x + shortcut)


def _bn(x, is_training,name):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta'+"_"+name,
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma'+"_"+name,
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean'+"_"+name,
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance'+"_"+name,
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, config.BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, config.BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, config.BN_EPSILON)

    return x


def _fc(x, units_out):

    num_units_in = x.get_shape()[1]
    num_units_out = units_out

    weights_initializer = tf.truncated_normal_initializer(
        stddev=config.FC_WEIGHT_STDDEV)

    weights = _get_variable(str(units_out)+'weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=config.FC_WEIGHT_STDDEV)
    biases = _get_variable(str(units_out)+'biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)

    return x


def _get_variable(name,
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
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def _conv(x, kernel,stride,filters_out,name):
    # ksize = c['ksize']
    # stride = c['stride']
    # filters_out = c['conv_filters_out']

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
    weights = _get_variable('weights'+"_"+name,
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=config.CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride[0], stride[1], 1], padding='SAME')


def _max_pool(x, ksize=1, stride=1):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID')


class VDCNN(object):
    """
    very deep CNN for text classification.
    """
    def __init__(
      self, sequence_length,
            num_classes,
            vocab_size,
            embedding_size,
            filter_sizes,
            num_filters,
            is_training,
            l2_reg_lambda=0.0):

        # vocab_size = 69# we use none atomic now
        # embedding_size = 16
        temp_kernel = (3, embedding_size)
        kernel = (3, 1)
        stride = (2, 1)
        # padding = (1, 0)
        # kmax = 8
        num_filters1 = 64
        num_filters2 = 128
        num_filters3 = 256
        num_filters4 = 512
        activation = tf.nn.relu
        fc1_hidden_size = 4096
        fc2_hidden_size = 2048
        num_output = 2

        self.is_training = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")#TODO char load embedding
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Temp Conv (in: batch, 1, 1014, 16)
        conv0 = _conv(x=self.embedded_chars_expanded, kernel=temp_kernel,
                      stride=stride, filters_out=num_filters1,name='conv0')
        act0 = activation(features=conv0, name='relu')

        # CONVOLUTION_BLOCK (1 of 4) -> 64 FILTERS
        with tf.variable_scope('block1'):
            conv11 = _conv(x=act0, kernel=kernel, stride=stride,filters_out=num_filters1,name='conv11')
            norm11 = _bn(conv11, self.is_training,name='norm11')
            act11 = activation(features=norm11, name='relu')
            conv12 = _conv(x=act11, kernel=kernel, stride=stride,filters_out=num_filters1,name='conv12')
            norm12 = _bn(conv12, self.is_training,name='norm12')
            act12 = activation(features=norm12, name='relu')

        # CONVOLUTION_BLOCK (2 of 4) -> 128 FILTERS
        with tf.variable_scope('block2'):
            conv61 = _conv(x=act12, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv61')
            norm61 = _bn(conv61, self.is_training, name='norm61')
            act61 = activation(features=norm61, name='relu')
            conv62 = _conv(x=act61, kernel=kernel, stride=stride, filters_out=num_filters2, name='conv62')
            norm62 = _bn(conv62, self.is_training,name='norm62')
            act62 = activation(features=norm62, name='relu')

        # CONVOLUTION_BLOCK (3 of 4) -> 256 FILTERS
        with tf.variable_scope('block3'):
            conv111 = _conv(x=act62, kernel=kernel, stride=stride, filters_out=num_filters3, name='conv111')
            norm111 = _bn(conv111, self.is_training,name='norm111')
            act111 = activation(features=norm111, name='relu')
            conv112 = _conv(x=act111, kernel=kernel, stride=stride, filters_out=num_filters3,name='conv112')
            norm112 = _bn(conv112, self.is_training, name='norm112')
            act112 = activation(features=norm112, name='relu')

        # CONVOLUTION_BLOCK (4 of 4) -> 512 FILTERS
        with tf.variable_scope('block4'):
            conv131 = _conv(x=act112, kernel=kernel, stride=stride, filters_out=num_filters4,name='conv131')
            norm131 = _bn(conv131, self.is_training,name='norm131')
            act131 = activation(features=norm131, name='relu')
            conv132 = _conv(x=act131, kernel=kernel, stride=stride, filters_out=num_filters4,name='conv132')
            norm132 = _bn(conv132, self.is_training,name='norm132')
            act132 = activation(features=norm132, name='relu')

        # K-max pooling (k=8) ? vs max pooling , according to paper,
        # Max-pooling performs better than other pool-
        # ing types. In terms of pooling, we can also see
        # that max-pooling performs best overall, very close
        # to convolutions with stride 2, but both are signifi-
        # cantly superior to k-max pooling.

        max_pool = _max_pool(act132)
        flatten = tf.reshape(max_pool, [-1, num_filters4])

        # Fully connected layers (fc)
        # fc1
        fc1 = _fc(flatten,fc1_hidden_size)
        act_fc1 = activation(fc1)

        # fc2
        fc2 = _fc(act_fc1,fc2_hidden_size)
        act_fc2 = activation(fc2)

        # fc3
        logits = _fc(act_fc2, num_output)
        predictions = tf.argmax(logits, 1, name="prediction")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


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


