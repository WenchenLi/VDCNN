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
from config import FEATURE_LEN
import tensorflow as tf
import numpy as np
import re

LABEL_start = '__label__'


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


def transform_sogou_data(data_file):
    """
    transform sogou data into fastText training format
    """
    examples = list(open(data_file, "r").readlines())
    with open("data/sogou_news_csv/sogou_data_train_dev.txt", 'w') as f:
        for e in examples:
            split_index = 3
            sentence = e[split_index+1:].strip()
            label = LABEL_start+e[:split_index]
            f.write(sentence + " " + label +"\n")


def load_data_and_labels_fasttext(data_file):
    """    
    
    :param data_file, expected each line is each sentence with it's label, label started with __label__ 
    :return: 
    """
    # Load data from files
    examples = list(open(data_file, "r").readlines())

    def get_strip_sentence_and_label_dict():
        index2label = []
        label2index = {}
        check_label_exists = {}
        current_label_index = 0

        labels = []
        examples_sentences = []

        for e in examples:
            label_start_index = e.find(LABEL_start)
            current_label = e[label_start_index:]
            labels.append(current_label)

            examples_sentences.append(e[:label_start_index].strip())

            if current_label not in check_label_exists:
                check_label_exists[current_label] = 1
                index2label.append(current_label)
                label2index[current_label] = current_label_index
                current_label_index += 1
            else:
                continue

        return examples_sentences, labels, index2label,label2index

    def onehot_encode(label):
        """
        encode label to one hot vector
        :param label: 
        :return: one hot vector
        """
        index = label2index[label]
        vector = [0] * num_unique_labels
        vector[index] = 1
        return vector

    examples_sentences, labels, index2label, label2index = get_strip_sentence_and_label_dict()
    num_unique_labels = len(index2label)

    # Split by words, trim to FEATURE_LEN chars
    x_text = examples_sentences
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [list(sentence.lower())[:FEATURE_LEN] for sentence in x_text]
    # Generate labels
    labels = [onehot_encode(l) for l in labels] #use one hot for each label
    y = np.array(labels)
    return [x_text, y,index2label]


def load_data_and_labels_change(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    with open("rt_data_all.txt", 'w') as f:


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
        return ckpt.model_checkpoint_path
    else:# todo error message
        pass

    # return None

if __name__=="__main__":
    # p= '/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt-polarity.pos'
    # n = '/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt-polarity.neg'
    # load_data_and_labels_change(p,n)
    # load_data_and_labels_fasttext("/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt_data_all.txt")
    transform_sogou_data("/home/wenchen/projects/VDCNN/data/sogou_news_csv/train.csv")