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

import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from litdata.streaming.item_loader import Interval
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


def _associate_chunks_and_internals_to_workers(
    distributed_env: _DistributedEnv,
    indexes: Any,
    chunk_intervals: List[Interval],
    drop_last: bool = False,
    num_workers: int = 1,
    batch_size: int = 1,
) -> Tuple[List[List[int]], List[Any]]:
    num_items = sum([(interval[2] - interval[1]) for interval in chunk_intervals])
    world_size = distributed_env.world_size * num_workers
    num_items_per_workers: List[int] = [
        num_items // world_size + num_items % world_size
        if rank == world_size - 1 and not drop_last
        else num_items // world_size
        for rank in range(world_size)
    ]
    if drop_last:
        num_items_per_workers = [batch_size * int(item // batch_size) for item in num_items_per_workers]

    chunks_per_workers: List[List[int]] = [[] for _ in range(world_size)]
    intervals_per_workers: List[List[List[int]]] = [[] for _ in range(world_size)]

    # 4. Assign the chunk & intervals to each rank
    for chunk_index, chunk_interval in zip(indexes, chunk_intervals):
        rank = 0

        while True:
            if rank == len(num_items_per_workers):
                break

            items_left_to_assign = num_items_per_workers[rank]

            if items_left_to_assign == 0:
                rank += 1
                continue

            items_in_chunk = chunk_interval[2] - chunk_interval[1]

            if items_in_chunk == 0:
                break

            if items_in_chunk > items_left_to_assign:
                chunks_per_workers[rank].append(chunk_index)

                chunk_start, chunk_roi_start, chunk_roi_end, chunk_end = chunk_interval

                intervals_per_workers[rank].append(
                    [chunk_start, chunk_roi_start, chunk_roi_start + items_left_to_assign, chunk_end]
                )
                chunk_interval = Interval(chunk_start, chunk_roi_start + items_left_to_assign, chunk_roi_end, chunk_end)
                num_items_per_workers[rank] = 0
                rank += 1
            else:
                chunks_per_workers[rank].append(chunk_index)
                intervals_per_workers[rank].append(list(chunk_interval))
                num_items_per_workers[rank] -= items_in_chunk
                break

    return chunks_per_workers, intervals_per_workers


def _find_chunks_per_workers_on_which_to_skip_deletion(
    num_workers: int,
    batch_size: int,
    workers_chunks: List[List[int]],
    workers_intervals: List[List[int]],
) -> Dict[int, List[int]]:
    # {1: [2, 3, 4, 5]}
    # [2, 3] belongs to rank 0
    # [4, 5] belongs to rank 1
    shared_chunks = _get_shared_chunks(workers_chunks)

    # workers_index_sharing_chunks
    # {1: (0, [2, 3], (1, [4, 5]))}
    shared_chunks_aggregated_by_rank = _aggregate_shared_chunks_per_rank(shared_chunks, num_workers)

    # breakpoint()

    max_trackers = {}
    to_disable = {}
    for chunk_index, map_local_rank_to_worker_ids in shared_chunks_aggregated_by_rank.items():
        for local_rank, workers_index_sharing_chunks_for_this_rank in map_local_rank_to_worker_ids.items():
            # get all the worker chunks and intervals for this distributed rank
            workers_slice = slice(local_rank * num_workers, (local_rank + 1) * num_workers)
            workers_chunks_for_this_rank = copy.deepcopy(workers_chunks[workers_slice])
            workers_intervals_for_this_rank = copy.deepcopy(  # TODO: rename
                [
                    [interval[2] - interval[1] for interval in worker_intervals]
                    for worker_intervals in workers_intervals[workers_slice]
                ]
            )

            num_shared_workers_for_this_rank = len(workers_index_sharing_chunks_for_this_rank)
            worker_tracker_idx = 0
            num_of_samples_to_carry_to_next_chunk = None
            counter = 0

            while True:
                chunks_of_currently_loaded_worker = workers_chunks_for_this_rank[worker_tracker_idx % num_workers]
                intervals_of_currently_loaded_worker = workers_intervals_for_this_rank[worker_tracker_idx % num_workers]
                if len(intervals_of_currently_loaded_worker) == 0:
                    worker_tracker_idx += 1
                    continue

                num_samples_left_for_this_worker_chunk = intervals_of_currently_loaded_worker[0]

                remover = (
                    batch_size
                    if num_of_samples_to_carry_to_next_chunk is None
                    else num_of_samples_to_carry_to_next_chunk
                )

                if num_samples_left_for_this_worker_chunk > remover:
                    # We have consumed a batch, going to the next worker
                    workers_intervals_for_this_rank[worker_tracker_idx % num_workers][0] -= remover
                    counter += remover
                    num_of_samples_to_carry_to_next_chunk = None
                else:
                    # We have consumed a batch, going to the next worker
                    current_worker_chunk_index = workers_chunks_for_this_rank[worker_tracker_idx % num_workers].pop(0)
                    workers_intervals_for_this_rank[worker_tracker_idx % num_workers].pop(0)
                    counter += remover

                    if current_worker_chunk_index == chunk_index:
                        num_shared_workers_for_this_rank -= 1
                        # breakpoint()

                    # We consumed entirely the chunk of the worker we were tracking, let's break
                    # TODO: Maybe, we can prevent loading over and over for each worker
                    if num_shared_workers_for_this_rank == 0 and current_worker_chunk_index == chunk_index:
                        if chunk_index not in max_trackers:
                            max_trackers[chunk_index] = (
                                local_rank * num_workers + worker_tracker_idx % num_workers,
                                counter,
                            )
                        else:
                            if max_trackers[chunk_index][1] < counter:
                                max_trackers[chunk_index] = (
                                    local_rank * num_workers + worker_tracker_idx % num_workers,
                                    counter,
                                )

                        break

                    if num_samples_left_for_this_worker_chunk != batch_size:
                        num_of_samples_to_carry_to_next_chunk = batch_size - num_samples_left_for_this_worker_chunk

                    if remover != batch_size:
                        num_of_samples_to_carry_to_next_chunk = None

                if num_of_samples_to_carry_to_next_chunk is None:
                    worker_tracker_idx += 1

            # else:
            #     # I don't know if this is possible
            #     break

    for chunk_index, worker_ids in shared_chunks.items():
        last_worker_idx = max_trackers[chunk_index][0]
        to_disable[chunk_index] = [worker_idx for worker_idx in worker_ids if worker_idx != last_worker_idx]
    return to_disable


def _get_shared_chunks(workers_chunks: List[List[int]]) -> Dict[int, List[int]]:
    shared_chunks = {}
    for worker, chunks in enumerate(workers_chunks):
        for chunk in chunks:
            if chunk not in shared_chunks:
                shared_chunks[chunk] = [worker]
            else:
                shared_chunks[chunk].append(worker)
    return {chunk: workers for chunk, workers in shared_chunks.items() if len(workers) > 1}


def _aggregate_shared_chunks_per_rank(shared_chunks, num_workers) -> Dict[int, List[int]]:
    aggregated_shared_chunks_per_rank = {}
    for chunk_index, workers_ids in shared_chunks.items():
        aggregated_shared_chunks_per_rank[chunk_index] = {}
        for worker_idx in workers_ids:
            if (worker_idx // num_workers) not in aggregated_shared_chunks_per_rank[chunk_index]:
                aggregated_shared_chunks_per_rank[chunk_index][worker_idx // num_workers] = []
            aggregated_shared_chunks_per_rank[chunk_index][worker_idx // num_workers].append(worker_idx)
    return aggregated_shared_chunks_per_rank


def _map_node_worker_rank_to_chunk_indexes_to_not_delete(to_disable):
    map_node_worker_rank_to_chunk_indexes = {}
    for chunk_index, worker_ids in to_disable.items():
        for worker_idx in worker_ids:
            if worker_idx not in map_node_worker_rank_to_chunk_indexes:
                map_node_worker_rank_to_chunk_indexes[worker_idx] = []
            map_node_worker_rank_to_chunk_indexes[worker_idx].append(chunk_index)
    return map_node_worker_rank_to_chunk_indexes
