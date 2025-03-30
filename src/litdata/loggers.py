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
from litdata.utilities.env import _DistributedEnv, _WorkerEnv

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
        LOG_FILE = os.getenv("LITDATA_LOG_FILE")
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
            "ts:%(created)s; logger_name:%(name)s; level:%(levelname)s; PID:%(process)d; TID:%(thread)d; %(message)s"
        )
        # ENV - f"{WORLD_SIZE, GLOBAL_RANK, NNODES, LOCAL_RANK, NODE_RANK}"
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Attach handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


def configure_logger():
    os.environ["LITDATA_LOG_FILE"] = f"litdata-{time.strftime('%Y-%m-%d-%H-%M')}.log"
    LitDataLogger("litdata")


def _get_log_msg(data: dict):
    log_msg = ""

    env_info_data = env_info()
    data.update(env_info_data)

    for key, value in data.items():
        log_msg += f"{key}: {value};"
    return log_msg


def env_info():
    dist_env = _DistributedEnv.detect()
    worker_env = _WorkerEnv.detect()  # will all threads read the same value if decorate this function with `@cache`

    return {
        "dist_world_size": dist_env.world_size,
        "dist_global_rank": dist_env.global_rank,
        "dist_num_nodes": dist_env.num_nodes,
        "worker_world_size": worker_env.world_size,
        "worker_rank": worker_env.rank,
    }


# -> Chrome tracing colors
#     url: https://chromium.googlesource.com/external/trace-viewer/+/bf55211014397cf0ebcd9e7090de1c4f84fc3ac0/tracing/tracing/ui/base/color_scheme.html

# # ------

# thread_state_iowait: {r: 182, g: 125, b: 143},
# thread_state_running: {r: 126, g: 200, b: 148},
# thread_state_runnable: {r: 133, g: 160, b: 210},
# ....
chrome_trace_colors = {
    "pink": "thread_state_iowait",
    "green": "thread_state_running",
    "light_blue": "thread_state_runnable",
    "light_gray": "thread_state_sleeping",
    "brown": "thread_state_unknown",
    "blue": "memory_dump",
    "gray": "generic_work",
    "dark_green": "good",
    "orange": "bad",
    "red": "terrible",
    "black": "black",
    "bright_blue": "rail_response",
    "bright_red": "rail_animate",
    "orange_yellow": "rail_idle",
    "teal": "rail_load",
    "dark_blue": "used_memory_column",
    "light_sky_blue": "older_used_memory_column",
    "medium_gray": "tracing_memory_column",
    "pale_yellow": "cq_build_running",
    "light_green": "cq_build_passed",
    "light_red": "cq_build_failed",
    "mustard_yellow": "cq_build_attempt_running",
    "neon_green": "cq_build_attempt_passed",
    "dark_red": "cq_build_attempt_failed",
}
