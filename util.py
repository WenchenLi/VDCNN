# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
from matplotlib import font_manager

import jieba
import pypinyin

LABEL_start = '__label__'


def word_seg(sentence, segmenter=" "):
    seg_list = jieba.cut(sentence, cut_all=False)
    # print("Default Mode: " + segmenter.join(seg_list))  # 精确模式
    return list(seg_list)


def word2pinyin(word):
    list_res = pypinyin.pinyin(word, style=pypinyin.TONE2, heteronym=False)
    if len(list_res) > 1:
        return "( " + " ".join([p[0] for p in list_res]) + " )"
    else:
        return list_res[0][0]


def sentence2pinyin(sentence):
    seg = word_seg(sentence)
    res = []
    for s in seg:
        res.append(word2pinyin(s))
    return " ".join(res)


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


def transform_sogou_data(data_file, output_filename):
    """
    transform sogou data into fastText training format
    """
    examples = list(open(data_file, "r").readlines())
    with open("data/sogou_news_csv/" + output_filename, 'w') as f:
        for e in examples:
            split_index = 3
            sentence = e[split_index + 1:].strip()
            label = LABEL_start + e[:split_index]
            f.write(sentence + " " + label + "\n")


def transform_lungutang(data_file, output_filename, print_stats=True):
    """
    transform lungutang data into fastText training format
    """
    from collections import defaultdict
    import csv
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    stats = defaultdict(int)
    sentence_len = []
    with open("data/lungutang/" + output_filename, 'w') as fw:
        with open(data_file, 'r') as fr:
            reader = csv.reader(fr)
            for i, row in enumerate(reader):
                if i == 0: continue  # skip header
                raw_label = row[0].strip()
                stats[raw_label] += 1
                label = LABEL_start + raw_label

                raw_sentence = row[1]
                sentence = str(sentence2pinyin(raw_sentence))
                sentence_len.append(len(sentence))
                # print i,raw_sentence
                fw.write(sentence + " " + label + "\n")

    if print_stats:
        for k in stats:
            print k, stats[k]
        print "total", sum([stats.values()])
        print 'mean char length', np.mean(sentence_len)


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
                index2label.append(current_label.replace("\n", ""))
                label2index[current_label] = current_label_index
                current_label_index += 1
            else:
                continue

        return examples_sentences, labels, index2label, label2index

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
    labels = [onehot_encode(l) for l in labels]  # use one hot for each label
    y = np.array(labels)
    return [x_text, y, index2label]


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
            f.write(e + " " + LABEL_start + "pos" + '\n')

        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        for e in negative_examples:
            f.write(e + " " + LABEL_start + "neg" + '\n')


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
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
    else:  # todo error message
        pass

        # return None


def draw_confusion_matrix(cm,step,train_path):
    # cm = [[2.38600000e+03, 3.00000000e+00, 1.70000000e+01, 3.00000000e+01,
    #       1.70000000e+01, 2.00000000e+00, 3.00000000e+00, 0.00000000e+00,
    #       1.92000000e+02, 1.00000000e+00, 9.00000000e+00, 4.00000000e+00,
    #       1.00000000e+00],
    #      [3.00000000e+01, 1.40000000e+01, 1.00000000e+00, 1.00000000e+01,
    #       3.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       2.50000000e+01, 0.00000000e+00, 2.00000000e+00, 0.00000000e+00,
    #       0.00000000e+00],
    #      [9.00000000e+00, 1.00000000e+00, 4.76000000e+02, 2.00000000e+00,
    #       1.40000000e+01, 3.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       1.78000000e+02, 0.00000000e+00, 1.20000000e+01, 0.00000000e+00,
    #       2.00000000e+00],
    #      [2.40000000e+01, 6.00000000e+00, 1.00000000e+01, 6.20000000e+01,
    #       1.40000000e+01, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    #       1.86000000e+02, 0.00000000e+00, 1.00000000e+01, 1.00000000e+01,
    #       0.00000000e+00],
    #      [2.60000000e+01, 0.00000000e+00, 1.20000000e+01, 7.00000000e+00,
    #       9.71000000e+02, 3.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       9.50000000e+01, 0.00000000e+00, 8.00000000e+00, 2.00000000e+00,
    #       3.00000000e+00],
    #      [1.00000000e+00, 0.00000000e+00, 7.00000000e+00, 3.00000000e+00,
    #       4.00000000e+00, 5.90000000e+01, 0.00000000e+00, 0.00000000e+00,
    #       2.38000000e+02, 0.00000000e+00, 4.10000000e+01, 0.00000000e+00,
    #       0.00000000e+00],
    #      [6.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       3.00000000e+00, 1.00000000e+00, 5.10000000e+01, 0.00000000e+00,
    #       5.60000000e+01, 0.00000000e+00, 1.10000000e+01, 0.00000000e+00,
    #       1.00000000e+00],
    #      [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.00000000e+00,
    #       0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 4.00000000e+00,
    #       1.50000000e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       0.00000000e+00],
    #      [7.30000000e+01, 4.00000000e+00, 5.40000000e+01, 3.20000000e+01,
    #       2.50000000e+01, 4.10000000e+01, 1.30000000e+01, 3.00000000e+00,
    #       7.67500000e+03, 1.00000000e+00, 3.18000000e+02, 4.00000000e+00,
    #       1.70000000e+01],
    #      [3.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
    #       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       4.00000000e+00, 1.00000000e+00, 3.00000000e+00, 0.00000000e+00,
    #       0.00000000e+00],
    #      [9.00000000e+00, 0.00000000e+00, 1.50000000e+01, 1.00000000e+00,
    #       1.00000000e+01, 2.10000000e+01, 1.00000000e+00, 0.00000000e+00,
    #       6.34000000e+02, 0.00000000e+00, 8.74000000e+02, 0.00000000e+00,
    #       5.00000000e+00],
    #      [7.00000000e+00, 0.00000000e+00, 6.00000000e+00, 6.00000000e+00,
    #       2.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #       2.20000000e+01, 0.00000000e+00, 0.00000000e+00, 1.90000000e+01,
    #       0.00000000e+00],
    #      [1.00000000e+00, 0.00000000e+00, 2.00000000e+00, 0.00000000e+00,
    #       2.00000000e+00, 1.00000000e+01, 0.00000000e+00, 0.00000000e+00,
    #       2.44000000e+02, 0.00000000e+00, 4.30000000e+01, 0.00000000e+00,
    #       3.40000000e+01]]

    # __label__qq广告 2665
    # __label__个人 85
    # __label__人名广告 697
    # __label__其他 324
    # __label__微信广告 1127
    # __label__敏感 353
    # __label__无意义 130
    # __label__昵称广告 25
    # __label__正常 8260
    # __label__网站链接广告 12
    # __label__脏话 1570
    # __label__问答广告 62
    # __label__风险 336
    # fontP = font_manager.FontProperties()
    # fontP.set_family('SimHei')
    # fontP.set_size(14)

    labels = ["qq广告","个人","人名广告","其他","微信广告","敏感","无意义","昵称广告","正常","网站链接广告","脏话"
              ,"问答广告","风险"]

    conf_arr = np.array(cm, dtype=float)
    norm_conf = np.array([ row/np.sum(row) for row in conf_arr])

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = norm_conf.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("%.2f" % norm_conf[x][y], xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLM'

    plt.xticks(range(width), [l.decode('utf-8') for l in alphabet[:width]], )
    plt.yticks(range(height),[l.decode('utf-8') for l in alphabet[:height]],)


    plt.savefig(train_path+'/confusion_matrix'+str(step)+'.png', format='png')
    for i in xrange(len(labels)):
        print alphabet[i],":",labels[i],"accuracy:",cm[i][i]/sum(cm[i])

if __name__ == "__main__":
    # p= '/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt-polarity.pos'
    # n = '/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt-polarity.neg'
    # load_data_and_labels_change(p,n)
    # load_data_and_labels_fasttext("/home/wenchen/projects/VDCNN/data/rt-polaritydata/rt_data_all.txt")
    # transform_sogou_data("/home/wenchen/projects/VDCNN/data/sogou_news_csv/train.csv","sogou_data_train_dev.txt")
    # transform_sogou_data("/home/wenchen/projects/VDCNN/data/sogou_news_csv/test.csv", "sogou_data_test.txt")
    # print word2pinyin("中心")
    # print sentence2pinyin("我来到北京清华大学")

    transform_lungutang("data/lungutang/lungutang_all_update_13.csv", "lungutang_13.txt")
    # confusion_matrix()