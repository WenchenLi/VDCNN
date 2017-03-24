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
special operations used in VDCNN,
k_max_pool
"""
import tensorflow as tf


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def _normalize_axis(axis, ndim):
    if type(axis) is tuple:
        axis = list(axis)
    if type(axis) is list:
        for i, a in enumerate(axis):
            if a is not None and a < 0:
                axis[i] = a % ndim
    else:
        if axis is not None and axis < 0:
            axis = axis % ndim
    return axis


def ndim(x):
    '''Returns the number of axes in a tensor, as an integer.
    '''
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def _sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    axis = _normalize_axis(axis, ndim(x))
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def categorical_crossentropy(output, target, from_logits=False):
    '''Categorical crossentropy between an output tensor
    and a target tensor, where the target is a tensor of the same
    shape as the output.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(output, target)


def customized_categorical_crossentropy(y_true, y_pred):
    # print "ce",K.sum(K.categorical_crossentropy(y_pred, y_true)).get_shape()
    # print K.categorical_crossentropy(y_pred, y_true).get_shape()
    return _sum(categorical_crossentropy(y_pred, y_true))


def concatenate(tensors, nm,axis=-1):
    '''Concantes a list of tensors alongside the specified axis.
    '''
    if axis < 0:
        if len(tensors[0].get_shape()):
            axis = axis % len(tensors[0].get_shape())
        else:
            axis = 0
    return tf.concat(axis, tensors,name=nm)


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID')


# class k_max_pool(mx.operator.CustomOp):
#
#     """
#     https://github.com/CNevd/DeepLearning-Mxnet/blob/master/DCNN/dcnn_train.py#L15
#     """
#
#     def __init__(self, k):
#         super(k_max_pool, self).__init__()
#         self.k = int(k)
#
#     def forward(self, is_train, req, in_data, out_data, aux):
#         x = in_data[0].asnumpy()
#         # assert(4 == len(x.shape))
#         ind = np.argsort(x, axis=2)
#         sorted_ind = np.sort(ind[:, :, -(self.k):, :], axis=2)
#         dim0, dim1, dim2, dim3 = sorted_ind.shape
#         self.indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
#         self.indices_dim1 = np.transpose(
#             np.arange(dim1).repeat(dim2 * dim3).reshape((dim1 * dim2 * dim3, 1)).repeat(dim0, axis=1)).flatten()
#         self.indices_dim2 = sorted_ind.flatten()
#         self.indices_dim3 = np.transpose(
#             np.arange(dim3).repeat(dim2).reshape((dim2 * dim3, 1)).repeat(dim0 * dim1, axis=1)).flatten()
#         y = x[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3].reshape(sorted_ind.shape)
#         self.assign(out_data[0], req[0], mx.nd.array(y))
#
#     def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
#         x = out_grad[0].asnumpy()
#         y = in_data[0].asnumpy()
#         # assert(4 == len(x.shape))
#         # assert(4 == len(y.shape))
#         y[:, :, :, :] = 0
#         y[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3] \
#             = x.reshape([x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3], ])
#         self.assign(in_grad[0], req[0], mx.nd.array(y))

#
# def conv(x, c):
#     ksize = c['ksize']
#     stride = c['stride']
#     filters_out = c['conv_filters_out']
#
#     filters_in = x.get_shape()[-1]
#     shape = [ksize, ksize, filters_in, filters_out]
#     initializer = tf.truncated_normal_initializer(stddev=config.CONV_WEIGHT_STDDEV)
#     weights = _get_variable('weights',
#                             shape=shape,
#                             dtype='float',
#                             initializer=initializer,
#                             weight_decay=config.CONV_WEIGHT_DECAY)
#     return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
# def fc(x, c, units_out):
#     num_units_in = x.get_shape()[1]
#     assert units_out in ['L', 'S'] or type(units_out) is int, \
#         "units_out badly configured, accepting int or ['L','S']"
#     if type(units_out) is int:
#         num_units_out = units_out
#     else:
#         num_units_out = c[units_out + '_fc_units_out']
#
#     weights_initializer = tf.truncated_normal_initializer(
#         stddev=config.FC_WEIGHT_STDDEV)
#
#     weights = _get_variable(str(units_out)+'weights',
#                             shape=[num_units_in, num_units_out],
#                             initializer=weights_initializer,
#                             weight_decay=config.FC_WEIGHT_STDDEV)
#     biases = _get_variable(str(units_out)+'biases',
#                            shape=[num_units_out],
#                            initializer=tf.zeros_initializer)
#     x = tf.nn.xw_plus_b(x, weights, biases)
#
#     return x

# def bn(x, c):
#     x_shape = x.get_shape()
#     params_shape = x_shape[-1:]
#
#     axis = list(range(len(x_shape) - 1))
#
#     beta = _get_variable('beta',
#                          params_shape,
#                          initializer=tf.zeros_initializer)
#     gamma = _get_variable('gamma',
#                           params_shape,
#                           initializer=tf.ones_initializer)
#
#     moving_mean = _get_variable('moving_mean',
#                                 params_shape,
#                                 initializer=tf.zeros_initializer,
#                                 trainable=False)
#     moving_variance = _get_variable('moving_variance',
#                                     params_shape,
#                                     initializer=tf.ones_initializer,
#                                     trainable=False)
#
#     # These ops will only be preformed when training.
#     mean, variance = tf.nn.moments(x, axis)
#     update_moving_mean = moving_averages.assign_moving_average(moving_mean,
#                                                                mean, config.BN_DECAY)
#     update_moving_variance = moving_averages.assign_moving_average(
#         moving_variance, variance, config.BN_DECAY)
#     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
#     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
#
#     mean, variance = control_flow_ops.cond(
#         c['is_training'], lambda: (mean, variance),
#         lambda: (moving_mean, moving_variance))
#
#     x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, config.BN_EPSILON)
#
#     return x

# @mx.operator.register("k_max_pool")
# class k_max_poolProp(mx.operator.CustomOpProp):
#     def __init__(self, k):
#         self.k = int(k)
#         super(k_max_poolProp, self).__init__(True)
#
#     def list_argument(self):
#         return ['data']
#
#     def list_outputs(self):
#         return ['output']
#
#     def infer_shape(self, in_shape):
#         data_shape = in_shape[0]
#         assert (len(data_shape) == 4)
#         out_shape = (data_shape[0], data_shape[1], self.k, data_shape[3])
#         return [data_shape], [out_shape]
#
#     def create_operator(self, ctx, shapes, dtypes):
#         return k_max_pool(self.k)