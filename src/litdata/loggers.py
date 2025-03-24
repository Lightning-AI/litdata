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

from litdata.constants import _DEBUG

logging.basicConfig(filename="trace.log", level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger(__name__)

debug_log = lambda msg: logger.debug(msg) if _DEBUG else None  # Inline-like
