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

FEATURE_LEN = 128
CHAR_EBD_SIZE = 16
BATCH_SIZE = 128
EPOCHS = 10
SD = 0.05  # std for gaussian distribution
NOUTPUT = 2
DATA_SHAPE = (BATCH_SIZE, 1, FEATURE_LEN, 1)
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
CHIENSE_ALPHABET = []#TODO add chiense common words here as atomic input elements

# resnet stuff
# FULLCHARS = [u'1', u'0', u'3', u'2', u'5', u'4', u'7', u'6', u'9', u'8', 'empty', u'A', u'C', u'B', u'E', u'D', u'G',
#              u'F', u'I', u'H', u'K', u'J', u'M', u'L', u'O', u'N', u'Q', u'P', u'S', u'R', u'U', u'T', u'W', u'V', u'Y',
#              u'X', u'Z', u'a', u'c', u'b', u'e', u'd', u'g', u'f', u'i', u'h', u'k', u'j', u'm', u'l', u'o', u'n', u'q',
#              u'p', u's', u'r', u'u', u't', u'w', u'v', u'y', u'x', u'z']
# NUM_CLASSES = 63
# CHAR2INDEX = {'1': 0, '0': 1, '3': 2, '2': 3, '5': 4, '4': 5, '7': 6, '6': 7, '9': 8, '8': 9, 'empty': 10, 'A': 11,
#               'C': 12, 'B': 13, 'E': 14, 'D': 15, 'G': 16, 'F': 17, 'I': 18, 'H': 19, 'K': 20, 'J': 21, 'M': 22,
#               'L': 23, 'O': 24, 'N': 25, 'Q': 26, 'P': 27, 'S': 28, 'R': 29, 'U': 30, 'T': 31, 'W': 32, 'V': 33,
#               'Y': 34, 'X': 35, 'Z': 36, 'a': 37, 'c': 38, 'b': 39, 'e': 40, 'd': 41, 'g': 42, 'f': 43, 'i': 44,
#               'h': 45, 'k': 46, 'j': 47, 'm': 48, 'l': 49, 'o': 50, 'n': 51, 'q': 52, 'p': 53, 's': 54, 'r': 55,
#               'u': 56, 't': 57, 'w': 58, 'v': 59, 'y': 60, 'x': 61, 'z': 62}
#
# MAX_LEN_CHARS = 6
# NUM_CHANNELS = 1
# RESNET_WINDOW_SHAPE = (200, 200)
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 0.001
REPORT_STEP = 1000 # prediction on the validation dataset, and then save the model
TRAIN_DIR = './save_model/' #where to save and load the model
DATA_DIR ="./data/train_50000" #where the data is
RESUME = True
TRAINING_STEPS = 100000
PER_PROCESS_GPU_MEMORY_FRACTION = .95

MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = .99
BN_EPSILON = 1e-6
CONV_WEIGHT_DECAY = 0.00000
CONV_WEIGHT_STDDEV = 0.01
FC_WEIGHT_DECAY = 0.00001
FC_WEIGHT_STDDEV = 0.0002
_EPSILON = 10e-8
VALIDATE_BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
