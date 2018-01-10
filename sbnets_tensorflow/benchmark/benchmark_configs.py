"""

   Sparse Blocks Network
   Copyright (c) 2017, Uber Technologies, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

#
# Benchmark common configs.
#
INPUT_SIZE_DICT = {
    'resnet-50': [
        [400, 700, 256, 64],    # YAPF_NO_FORMAT
        [200, 350, 512, 128],
        [100, 175, 1024, 256],
        [50, 88, 2048, 512],
    ],
    'resnet-v2': [
        [400, 700, 96, 24],    # YAPF_NO_FORMAT
        [200, 350, 192, 48],
        [100, 175, 256, 64],
        [50, 88, 384, 96],
    ],
    'resnet-v3': [
        [400, 700, 128, 32],    # YAPF_NO_FORMAT
        [200, 350, 256, 64],
        [100, 175, 384, 96],
        [50, 88, 512, 128],
    ]
}
SPARSITY_LIST = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
SPARSITY_LIST = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
