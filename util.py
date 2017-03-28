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
utilities used in paper VDCNN
"""
import wget
import os
import pandas as pd
import functools
import threading
import Queue
from config import FEATURE_LEN, BATCH_SIZE
import tensorflow as tf

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels_fasttext(data_file):
    """    
    
    :param data_file_path, expected each line is each sentence with it's label, label started with __label__ 
    :return: 
    """
    # Load data from files

    examples = list(open(data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    # Split by words, trim to FEATURE_LEN chars
    x_text = examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [list(sentence.lower())[:FEATURE_LEN] for sentence in x_text]
    # Generate labels
    labels = [[1, 0] for _ in examples] #use one hot for each label
    y = np.concatenate([labels, labels], 0)
    return [x_text, y]


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words, trim to FEATURE_LEN chars
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [list(sentence.lower())[:FEATURE_LEN] for sentence in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_data_and_labels_change(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    with open("rt_data_all.txt", 'w') as f:
        LABEL_start = '__label__'

        # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        for e in positive_examples:
            f.write(e + " "+ LABEL_start+"pos"+'\n')

        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        for e in negative_examples:
            f.write(e + " "+ LABEL_start+"neg"+'\n')


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def latest_checkpoint(checkpoint_dir, latest_filename=None):
    """Finds the filename of latest saved checkpoint file.
    
    Args:
      checkpoint_dir: Directory where the variables were saved.
      latest_filename: Optional name for the protocol buffer file that
        contains the list of most recent checkpoint filenames.
        See the corresponding argument to `Saver.save()`.
    
    Returns:
      The full path to the latest checkpoint or `None` if no checkpoint was found.
    """
    # Pick the latest checkpoint based on checkpoint state.
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir, latest_filename)
    if ckpt and ckpt.model_checkpoint_path:
        # Look for either a V2 path or a V1 path, with priority for V2.

        # v1_path = tf.train._prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
        #                                      saver_pb2.SaverDef.V1)
        return ckpt.model_checkpoint_path
    else:# todo error message
        pass

    # return None

if __name__=="__main__":
    p= '/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt-polarity.pos'
    n = '/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt-polarity.neg'
    load_data_and_labels_change(p,n)