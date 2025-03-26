# Copyright The Lightning AI team.
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

import logging
import os
import sys
import time

from litdata.constants import _DEBUG

# Create the root logger for the library
root_logger = logging.getLogger("litdata")


def get_logger_level(level: str) -> int:
    """Get the log level from the level string."""
    level = level.upper()
    if level in logging._nameToLevel:
        return logging._nameToLevel[level]
    raise ValueError(f"Invalid log level: {level}. Valid levels: {list(logging._nameToLevel.keys())}.")


class LitDataLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.log_file, self.log_level = self.get_log_file_and_level()
        self.setup_logger()

    @staticmethod
    def get_log_file_and_level():
        LOG_FILE = os.getenv("LITDATA_LOG_FILE", f"litdata-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log")
        LOG_LEVEL = os.getenv("LITDATA_LOG_LEVEL", "INFO" if not _DEBUG else "DEBUG")

        LOG_LEVEL = get_logger_level(LOG_LEVEL)

        return LOG_FILE, LOG_LEVEL

    def setup_logger(self):
        """Configures logging by adding handlers and formatting."""
        if len(self.logger.handlers) > 0:  # Avoid duplicate handlers
            return

        self.logger.setLevel(self.log_level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)

        # Log format
        formatter = logging.Formatter(
            "time:%(asctime)s; name:%(name)s; level:%(levelname)s; PID:%(process)d; TID:%(thread)d; %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
        )
        # ENV - f"{WORLD_SIZE, GLOBAL_RANK, NNODES, LOCAL_RANK, NODE_RANK}"
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Attach handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


def configure_logger():
    LitDataLogger("litdata")
