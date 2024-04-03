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

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _pack_greedily(items: List[Any], weights: List[int], num_bins: int) -> Tuple[Dict[int, List[Any]], Dict[int, int]]:
    """Greedily pack items with given weights into bins such that the total weight of each bin is roughly equally
    distributed among all bins."""

    if len(items) != len(weights):
        raise ValueError(f"Items and weights must have the same length, got {len(items)} and {len(weights)}.")
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive.")

    sorted_items_and_weights = sorted(zip(items, weights), key=lambda x: x[1], reverse=True)
    bin_contents = defaultdict(list)
    bin_weights = {i: 0 for i in range(num_bins)}

    for item, weight in sorted_items_and_weights:
        min_bin_id = min(bin_weights, key=(lambda x: bin_weights[x]), default=0)
        bin_contents[min_bin_id].append(item)
        bin_weights[min_bin_id] += weight

    return bin_contents, bin_weights
