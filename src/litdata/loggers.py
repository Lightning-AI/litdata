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


def setup_logging(logger: logging.Logger):
    """Configures logging by adding handlers and formatting."""
    if len(logger.handlers) > 0:  # Avoid duplicate handlers
        return

    LOG_FILE = os.getenv("LITDATA_LOG_FILE", f"litdata-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log")
    LOG_LEVEL = os.getenv("LITDATA_LOG_LEVEL", "INFO" if not _DEBUG else "DEBUG")

    LOG_LEVEL = get_logger_level(LOG_LEVEL)

    logger.setLevel(LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(LOG_LEVEL)

    # Log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def get_logger_level(level: str) -> int:
    """Get the log level from the level string."""
    level = level.upper()
    if level in logging._nameToLevel:
        return logging._nameToLevel[level]
    raise ValueError(f"Invalid log level: {level}. Valid levels: {list(logging._nameToLevel.keys())}.")


# Apply handlers using the setup function
setup_logging(root_logger)
