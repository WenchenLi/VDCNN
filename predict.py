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
run the training/dev/testing
"""

# from util import
import numpy as np
import config
import sys
import tensorflow as tf
import util
from model import VDCNN
import os
from tensorflow.contrib import learn
import pickle
from util import sentence2pinyin


class VDCNN_model(object):
    """
        model wrapper similar to keras.model
    """

    def __init__(self, num_class, model_weights_dir, num_channel=1, device="gpu", device_id=0, variable_reuse=None,
                 is_chinese=False):
        """

        :param model_weights_dir: string, save_model dir
        :param device: string, cpu or gpu
        :param device_id: int,
        :param variable_reuse: bool, whether to reuse variable during prediction,(for multiple gpus, see below examples)
        """
        self.model_weights_dir = model_weights_dir
        self.per_process_gpu_memory_fraction = .95
        self.num_channel = num_channel
        self.device = device
        self.device_id = device_id
        self.variable_reuse = variable_reuse
        self.num_class = num_class
        self.is_chinese = is_chinese
        # load vocab

        self.vocabulary = learn.preprocessing.CategoricalVocabulary()
        for token in config.ALPHABET:
            self.vocabulary.add(token)
        self.vocabulary.freeze()
        self.index2label = pickle.load(open(os.path.join(
            self.model_weights_dir[:-11], 'index2label.pk'), 'rb'))

        max_document_length = config.FEATURE_LEN
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, vocabulary=self.vocabulary,
                                                                       tokenizer_fn=list)
        self.is_training = tf.placeholder('bool', [], name='is_training')
        # load model
        with tf.device(self.device + ":" + str(self.device_id)):
            self.model = VDCNN(
                feature_len=config.FEATURE_LEN,
                num_classes=num_class,
                vocab_size=70,  # fixed to 70, <unk> + 69 char in config
                embedding_size=config.CHAR_EBD_SIZE,
                is_training=self.is_training,
                depth=9)

        # Write vocabulary

        self.model_session = self.model_session()

    def model_session(self):
        """
        load a model with a tf session
        :return: a tf session
        """
        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(init)

        # resume
        latest = str(util.latest_checkpoint(self.model_weights_dir))
        if not latest:
            print "No checkpoint to continue from in", latest
            sys.exit(1)
        saver.restore(sess, latest)
        print "model loaded", latest

        return sess

    def predict(self, x_feed):
        """
        wrapper similar to scikit learn pred
        :param x_feed: sentences as list of str
        :return: list of strings, prediction result
        """
        if self.is_chinese:
            x_feed = [sentence2pinyin(s) for s in x_feed]
        x = np.array(list(self.vocab_processor.fit_transform(x_feed)))
        y_unit_placeholder = [0] * self.num_class
        feed_dict = {
            self.model.input_x: x,
            self.model.input_y: np.array(len(x)*[y_unit_placeholder]),#just a placeholder filller
            self.model.is_training: False
        }
        logits = self.model_session.run(
            self.model.logits, feed_dict)

        res = np.argmax(logits,axis=1)
        res = [self.index2label[i] for i in res]
        return res


if __name__ == '__main__':
    # load model parameters
    vdcnn = VDCNN_model(num_class=2,
                        model_weights_dir='/home/wenchen/projects/VDCNN/train_dir/1490994156/checkpoints',
                        num_channel=1, device="gpu", device_id=0, variable_reuse=None,is_chinese=False)

    sentences = ["a romantic comedy enriched by a sharp eye for manners and mores .","as pedestrian as they come ."]
    res = vdcnn.predict(sentences)
    print res
