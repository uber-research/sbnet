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
# Logging utility.
#

import datetime
import logging


class _MyFormatter(logging.Formatter):
    width = 24
    datefmt = '%Y-%m-%d %H:%M:%S.%f'

    def format(self, record):
        cpath = '%s:%s:%s' % (record.module, record.funcName, record.lineno)
        if len(cpath) > self.width:
            cpath = "..." + cpath[-self.width + 3:]
        record.message = record.getMessage()
        s = "{} {}: {}: {}".format(record.levelname,
                                   datetime.datetime.now().isoformat(chr(32)), cpath,
                                   record.getMessage())
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        return s


logger = None


def get():
    global logger
    if logger is None:
        logging.addLevelName(logging.WARNING,
                             "\033[0;33m%s:\033[1;0m" % logging.getLevelName(logging.WARNING))
        logging.addLevelName(logging.ERROR,
                             "\033[0;31m%s:\033[1;0m" % logging.getLevelName(logging.ERROR))
        logging.addLevelName(logging.INFO,
                             "\033[0;32m%s:\033[1;0m" % logging.getLevelName(logging.INFO))
        logging.addLevelName(logging.DEBUG,
                             "\033[0;39m%s:\033[1;0m" % logging.getLevelName(logging.DEBUG))
        logger = logging.getLogger(__name__)
        logger.propagate = False
        ch = logging.StreamHandler()
        formatter = _MyFormatter()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def main():
    log = get()
    log.setLevel(logging.DEBUG)
    log.debug("Hey")
    log.info("Hey")
    log.warning("Hey")
    log.error("Hey")


if __name__ == "__main__":
    main()
