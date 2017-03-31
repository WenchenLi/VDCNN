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
# BATCH_SIZE = 64
EPOCHS = 10
SD = 0.05  # std for gaussian distribution
NOUTPUT = 2
# DATA_SHAPE = (BATCH_SIZE, 1, FEATURE_LEN, 1)
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
CHINESE_ALPHABET = []#TODO add chiense common words here as atomic input elements



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
# VALIDATE_BATCH_SIZE = 128
# TEST_BATCH_SIZE = 128
