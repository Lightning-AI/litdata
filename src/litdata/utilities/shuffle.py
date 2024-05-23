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

from typing import Any, List, Tuple

import numpy as np

from litdata.utilities.env import _DistributedEnv


def _intra_node_chunk_shuffle(
    distributed_env: _DistributedEnv,
    chunks_per_ranks: List[List[int]],
    seed: int,
    current_epoch: int,
) -> List[int]:
    chunk_indexes_per_nodes: Any = [[] for _ in range(distributed_env.num_nodes)]
    process_per_node = distributed_env.world_size // distributed_env.num_nodes
    for rank, chunks_per_rank in enumerate(chunks_per_ranks):
        chunk_indexes_per_nodes[0 if distributed_env.num_nodes == 1 else rank // process_per_node].extend(
            chunks_per_rank
        )

    # shuffle the chunks associated to the node
    for i in range(len(chunk_indexes_per_nodes)):
        # permute the indexes within the node
        chunk_indexes_per_nodes[i] = np.random.RandomState(seed=seed + current_epoch).permutation(
            chunk_indexes_per_nodes[i]
        )

    return [index for chunks in chunk_indexes_per_nodes for index in chunks]


def _associate_chunks_and_internals_to_ranks(
    distributed_env: _DistributedEnv,
    indexes: Any,
    chunk_intervals: Any,
    drop_last: bool,
    num_workers: int = 1,
    batch_size: int = 1,
) -> Tuple[List[List[int]], List[Any]]:
    num_items = sum([(interval[-1] - interval[0]) for interval in chunk_intervals])
    num_items_per_ranks: List[int] = [
        num_items // distributed_env.world_size + num_items % distributed_env.world_size
        if rank == distributed_env.world_size - 1 and not drop_last
        else num_items // distributed_env.world_size
        for rank in range(distributed_env.world_size)
    ]
    ratio = num_workers * batch_size
    num_items_per_ranks = [ratio * int(item // ratio) for item in num_items_per_ranks]

    chunks_per_ranks: List[List[int]] = [[] for _ in range(distributed_env.world_size)]
    intervals_per_ranks: List[List[List[int]]] = [[] for _ in range(distributed_env.world_size)]

    # 4. Assign the chunk & intervals to each rank
    for chunk_index, chunk_interval in zip(indexes, chunk_intervals):
        rank = 0

        while True:
            if rank == len(num_items_per_ranks):
                break

            items_left_to_assign = num_items_per_ranks[rank]

            if items_left_to_assign == 0:
                rank += 1
                continue

            items_in_chunk = chunk_interval[-1] - chunk_interval[0]

            if items_in_chunk == 0:
                break

            if items_in_chunk > items_left_to_assign:
                chunks_per_ranks[rank].append(chunk_index)
                begin, end = chunk_interval
                intervals_per_ranks[rank].append([begin, begin + items_left_to_assign])
                chunk_interval = (begin + items_left_to_assign, end)
                num_items_per_ranks[rank] = 0
                rank += 1
            else:
                chunks_per_ranks[rank].append(chunk_index)
                intervals_per_ranks[rank].append(chunk_interval)
                num_items_per_ranks[rank] -= items_in_chunk
                break

    return chunks_per_ranks, intervals_per_ranks
