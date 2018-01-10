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
# Sparse convolution benchmark utility.
#
import os
import time
uname = os.environ['USER']


def prefix_path(x):
    """Gets home folder path."""
    return os.path.join('/home', uname, x)


def append_result(out_file, r):
    """Appends results to the output file."""
    print(r)
    rdict = r._asdict()
    keys = rdict.keys()
    with open(out_file, 'a') as f:
        f.write(','.join([str(rdict[ff]) for ff in keys]) + '\n')


def create_result(out_file, fields):
    """Creates output file and CSV header."""
    with open(out_file, 'w') as f:
        f.write(','.join(fields) + '\n')


def get_out_filename(prefix='sparse_conv_perf_out'):
    """Generates output filename."""
    out_file = prefix_path('{}_{:d}.csv'.format(prefix, int(time.time())))
    return out_file
