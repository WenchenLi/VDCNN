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
FEATURE_LEN = 256 # char length of the sentence, you can configure this to fit your problem 80% 128,
CHAR_EBD_SIZE = 16
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")

PER_PROCESS_GPU_MEMORY_FRACTION = .95
MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = .99
BN_EPSILON = 1e-6
CONV_WEIGHT_DECAY = 0.00000
CONV_WEIGHT_STDDEV = 0.01
FC_WEIGHT_DECAY = 0.00001
FC_WEIGHT_STDDEV = 0.0002
_EPSILON = 10e-8

