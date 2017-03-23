"""
Replicate "Very Deep Convolutional Networks for Natural Language Processing" by Alexis Conneau,
Holger Schwenk, Yann Le Cun, Loic Barraeau, 2016

New NLP architecture:
1. Operate at lowest atomic representation of text (characters)
2. Use deep-stack of local operations to learn high-level hierarchical representation
"""

import numpy as np
import time
import functools
import threading
import Queue
import pickle


ctx = mx.gpu(0)
AZ_ACC = "amazonsentimenik"
AZ_CONTAINER = "textclassificationdatasets"
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
FEATURE_LEN = 1014
BATCH_SIZE = 128
EPOCHS = 10
SD = 0.05  # std for gaussian distribution
NOUTPUT = 2
DATA_SHAPE = (BATCH_SIZE, 1, FEATURE_LEN, 1)





def create_vdcnn():
    """
    29 Convolutional Layers

    We want to increase the number of conv layers to 29 in the following structure:
    1 | 10 | 10 | 4 | 4 -> 4.6 million params

    We down-sample using convolutions with stride=2

    ToDo:
    2. Temporal batch norm vs. batch norm? -> "Temp batch norm applies same kind of regularization
    as batch norm, except that the activations in a mini-batch are jointly normalized over temporal
    instead of spatial locations"
    3. Double check that optional shortcuts are not used for the smaller nets (only for 49 conv layer one,
    as they reduce performance for 9, 17, 29 conv. layer models)
    """

    vocab_size = 69
    embedding_size = 16
    temp_kernel = (3, embedding_size)
    kernel = (3, 1)
    stride = (2, 1)
    padding = (1, 0)
    kmax = 8
    num_filters1 = 64
    num_filters2 = 128
    num_filters3 = 256
    num_filters4 = 512

    input_x = mx.sym.Variable('data')  # placeholder for input
    input_y = mx.sym.Variable('softmax_label')  # placeholder for output

    # Lookup Table 16
    embed_layer = mx.symbol.Embedding(
        data=input_x, input_dim=vocab_size, output_dim=embedding_size, name='word_embedding')
    embed_out = mx.sym.Reshape(
        data=embed_layer, shape=(BATCH_SIZE, 1, FEATURE_LEN, embedding_size))

    # Temp Conv (in: batch, 1, 1014, 16)
    conv0 = mx.symbol.Convolution(
        data=embed_out, kernel=temp_kernel, pad=padding, num_filter=num_filters1)
    act0 = mx.symbol.Activation(
        data=conv0, act_type='relu')

    # CONVOLUTION_BLOCK (1 of 4) -> 64 FILTERS
    # 10 Convolutional Layers
    conv11 = mx.symbol.Convolution(
        data=act0, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm11 = mx.symbol.BatchNorm(
        data=conv11)
    act11 = mx.symbol.Activation(
        data=norm11, act_type='relu')
    conv12 = mx.symbol.Convolution(
        data=act11, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm12 = mx.symbol.BatchNorm(
        data=conv12)
    act12 = mx.symbol.Activation(
        data=norm12, act_type='relu')

    conv21 = mx.symbol.Convolution(
        data=act12, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm21 = mx.symbol.BatchNorm(
        data=conv21)
    act21 = mx.symbol.Activation(
        data=norm21, act_type='relu')
    conv22 = mx.symbol.Convolution(
        data=act21, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm22 = mx.symbol.BatchNorm(
        data=conv22)
    act22 = mx.symbol.Activation(
        data=norm22, act_type='relu')

    conv31 = mx.symbol.Convolution(
        data=act22, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm31 = mx.symbol.BatchNorm(
        data=conv31)
    act31 = mx.symbol.Activation(
        data=norm31, act_type='relu')
    conv32 = mx.symbol.Convolution(
        data=act31, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm32 = mx.symbol.BatchNorm(
        data=conv32)
    act32 = mx.symbol.Activation(
        data=norm32, act_type='relu')

    conv41 = mx.symbol.Convolution(
        data=act32, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm41 = mx.symbol.BatchNorm(
        data=conv41)
    act41 = mx.symbol.Activation(
        data=norm41, act_type='relu')
    conv42 = mx.symbol.Convolution(
        data=act41, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm42 = mx.symbol.BatchNorm(
        data=conv42)
    act42 = mx.symbol.Activation(
        data=norm42, act_type='relu')

    conv51 = mx.symbol.Convolution(
        data=act42, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm51 = mx.symbol.BatchNorm(
        data=conv51)
    act51 = mx.symbol.Activation(
        data=norm51, act_type='relu')
    conv52 = mx.symbol.Convolution(
        data=act51, kernel=kernel, pad=padding, num_filter=num_filters1)
    norm52 = mx.symbol.BatchNorm(
        data=conv52)
    act52 = mx.symbol.Activation(
        data=norm52, act_type='relu')

    # CONVOLUTION_BLOCK (2 of 4) -> 128 FILTERS
    # 10 Convolutional Layers

    # First down-sampling
    conv61 = mx.symbol.Convolution(
        data=act52, kernel=kernel, pad=padding, stride=stride, num_filter=num_filters2)

    norm61 = mx.symbol.BatchNorm(
        data=conv61)
    act61 = mx.symbol.Activation(
        data=norm61, act_type='relu')
    conv62 = mx.symbol.Convolution(
        data=act61, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm62 = mx.symbol.BatchNorm(
        data=conv62)
    act62 = mx.symbol.Activation(
        data=norm62, act_type='relu')

    conv71 = mx.symbol.Convolution(
        data=act62, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm71 = mx.symbol.BatchNorm(
        data=conv71)
    act71 = mx.symbol.Activation(
        data=norm71, act_type='relu')
    conv72 = mx.symbol.Convolution(
        data=act71, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm72 = mx.symbol.BatchNorm(
        data=conv72)
    act72 = mx.symbol.Activation(
        data=norm72, act_type='relu')

    conv81 = mx.symbol.Convolution(
        data=act72, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm81 = mx.symbol.BatchNorm(
        data=conv81)
    act81 = mx.symbol.Activation(
        data=norm81, act_type='relu')
    conv82 = mx.symbol.Convolution(
        data=act81, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm82 = mx.symbol.BatchNorm(
        data=conv82)
    act82 = mx.symbol.Activation(
        data=norm82, act_type='relu')

    conv91 = mx.symbol.Convolution(
        data=act82, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm91 = mx.symbol.BatchNorm(
        data=conv91)
    act91 = mx.symbol.Activation(
        data=norm91, act_type='relu')
    conv92 = mx.symbol.Convolution(
        data=act91, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm92 = mx.symbol.BatchNorm(
        data=conv92)
    act92 = mx.symbol.Activation(
        data=norm92, act_type='relu')

    conv101 = mx.symbol.Convolution(
        data=act92, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm101 = mx.symbol.BatchNorm(
        data=conv101)
    act101 = mx.symbol.Activation(
        data=norm101, act_type='relu')
    conv102 = mx.symbol.Convolution(
        data=act101, kernel=kernel, pad=padding, num_filter=num_filters2)
    norm102 = mx.symbol.BatchNorm(
        data=conv102)
    act102 = mx.symbol.Activation(
        data=norm102, act_type='relu')


    # CONVOLUTION_BLOCK (3 of 4) -> 256 FILTERS
    # 4 Convolutional Layers

    # Second down-sampling
    conv111 = mx.symbol.Convolution(
        data=act102, kernel=kernel, pad=padding, stride=stride, num_filter=num_filters3)

    norm111 = mx.symbol.BatchNorm(
        data=conv111)
    act111 = mx.symbol.Activation(
        data=norm111, act_type='relu')
    conv112 = mx.symbol.Convolution(
        data=act111, kernel=kernel, pad=padding, num_filter=num_filters3)
    norm112 = mx.symbol.BatchNorm(
        data=conv112)
    act112 = mx.symbol.Activation(
        data=norm112, act_type='relu')

    conv121 = mx.symbol.Convolution(
        data=act112, kernel=kernel, pad=padding, num_filter=num_filters3)
    norm121 = mx.symbol.BatchNorm(
        data=conv121)
    act121 = mx.symbol.Activation(
        data=norm121, act_type='relu')
    conv122 = mx.symbol.Convolution(
        data=act121, kernel=kernel, pad=padding, num_filter=num_filters3)
    norm122 = mx.symbol.BatchNorm(
        data=conv122)
    act122 = mx.symbol.Activation(
        data=norm122, act_type='relu')

    # CONVOLUTION_BLOCK (4 of 4) -> 512 FILTERS
    # 4 Convolutional Layers

    # Third down-sampling
    conv131 = mx.symbol.Convolution(
        data=act122, kernel=kernel, pad=padding, stride=stride, num_filter=num_filters4)

    norm131 = mx.symbol.BatchNorm(
        data=conv131)
    act131 = mx.symbol.Activation(
        data=norm131, act_type='relu')
    conv132 = mx.symbol.Convolution(
        data=act131, kernel=kernel, pad=padding, num_filter=num_filters4)
    norm132 = mx.symbol.BatchNorm(
        data=conv132)
    act132 = mx.symbol.Activation(
        data=norm132, act_type='relu')

    conv141 = mx.symbol.Convolution(
        data=act132, kernel=kernel, pad=padding, num_filter=num_filters4)
    norm141 = mx.symbol.BatchNorm(
        data=conv141)
    act141 = mx.symbol.Activation(
        data=norm141, act_type='relu')
    conv142 = mx.symbol.Convolution(
        data=act141, kernel=kernel, pad=padding, num_filter=num_filters4)
    norm142 = mx.symbol.BatchNorm(
        data=conv142)
    act142 = mx.symbol.Activation(
        data=norm142, act_type='relu')

    # K-max pooling (k=8)
    kpool = mx.symbol.Custom(
        data=act142, op_type='k_max_pool', k=kmax)

    # Flatten (dimensions * feature length * filters)
    flatten = mx.symbol.Flatten(data=kpool)

    # First fully connected
    fc1 = mx.symbol.FullyConnected(
        data=flatten, num_hidden=4096)
    act_fc1 = mx.symbol.Activation(
        data=fc1, act_type='relu')
    # Second fully connected
    fc2 = mx.symbol.FullyConnected(
        data=act_fc1, num_hidden=2048)
    act_fc2 = mx.symbol.Activation(
        data=fc2, act_type='relu')
    # Third fully connected
    fc3 = mx.symbol.FullyConnected(
        data=act_fc2, num_hidden=NOUTPUT)
    net = mx.symbol.SoftmaxOutput(
        data=fc3, label=input_y, name="softmax")

    #Debug:
    arg_shape, output_shape, aux_shape = net.infer_shape(data=(DATA_SHAPE))
    print("Arg Shape: ", arg_shape)
    print("Output Shape: ", output_shape)
    print("Aux Shape: ", aux_shape)
    print("Created network")

    return net







