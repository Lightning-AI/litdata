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
) -> Tuple[List[List[int]], List[Any]]:
    num_items = sum([(interval[-1] - interval[0]) for interval in chunk_intervals])
    num_items_per_ranks: List[int] = [
        num_items // distributed_env.world_size + num_items % distributed_env.world_size
        if rank == distributed_env.world_size - 1 and not drop_last
        else num_items // distributed_env.world_size
        for rank in range(distributed_env.world_size)
    ]
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


def _find_rank_actions_for_shared_chunks(chunks_per_ranks: List[List[int]], intervals_per_ranks: List[Any]):
    num_ranks = len(chunks_per_ranks)

    shared_chunks_map = {}
    for rank, chunks in enumerate(chunks_per_ranks):
        intervals = intervals_per_ranks[rank]
        cum_intervals = np.cumsum([0] + [interval[1] - interval[0] for interval in intervals])
        for chunk_index, chunk in enumerate(chunks):
            if chunk not in shared_chunks_map:
                shared_chunks_map[chunk] = []
            shared_chunks_map[chunk].append([rank, cum_intervals[chunk_index], cum_intervals[chunk_index + 1]])

    rank_actions_download = {}
    rank_actions_delete = {}

    rank_actions_disable_download = {}
    rank_actions_disable_delete = {}

    for chunk_index, associations in shared_chunks_map.items():
        if len(associations) == 1:
            continue

        start_using = [v[1] for v in associations]
        stop_using = [v[2] for v in associations]

        # find the min(s)
        min_start_using = np.min(start_using)
        for v in associations:
            if v[1] == min_start_using:
                if v[0] not in rank_actions_download:
                    rank_actions_download[v[0]] = [chunk_index]
                else:
                    rank_actions_download[v[0]].append(chunk_index)


        max_stop_using = np.max(stop_using)
        for v in associations:
            if v[2] == max_stop_using:
                rank_actions_delete[v[0]] = [chunk_index]
                break


    for chunk_index, associations in shared_chunks_map.items():
        if len(associations) == 1:
            continue

        ranks = [v[0] for v in associations]

        for rank in ranks:
            if rank not in rank_actions_download:
                if rank not in rank_actions_disable_download:
                    rank_actions_disable_download[rank] = [chunk_index]
                else:
                    rank_actions_disable_download[rank].push(chunk_index)

            if rank not in rank_actions_delete:
                if rank not in rank_actions_disable_delete:
                    rank_actions_disable_delete[rank] = [chunk_index]
                else:
                    rank_actions_disable_delete[rank].push(chunk_index)

    return shared_chunks_map, rank_actions_disable_download, rank_actions_disable_delete