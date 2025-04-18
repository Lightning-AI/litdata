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

import os
from typing import Callable, Optional

import torch
from torch.utils.data import get_worker_info as torch_get_worker_info


class _DistributedEnv:
    """The environment of the distributed training.

    Args:
        world_size: The number of total distributed training processes
        global_rank: The rank of the current process within this pool of training processes
        num_nodes: The number of nodes used for distributed training
            (e.g. the number of GPUs(devices) per node * the number of nodes = world_size)

    """

    def __init__(self, world_size: int, global_rank: int, num_nodes: int):
        self.world_size = world_size
        self.global_rank = global_rank
        self.num_nodes = num_nodes

    @classmethod
    def detect(cls) -> "_DistributedEnv":
        """Tries to automatically detect the distributed environment parameters.

        .. note::
            This detection may not work in processes spawned from the distributed processes (e.g. DataLoader workers)
            as the distributed framework won't be initialized there.
            It will default to 1 distributed process in this case.

        """
        if _is_in_map_or_optimize():
            return cls._instantiate_in_map_or_optimize()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
            # Note: On multi node CPU, the number of nodes won't be correct.
            if torch.cuda.is_available() and world_size // torch.cuda.device_count() >= 1:
                num_nodes = world_size // torch.cuda.device_count()
            else:
                num_nodes = 1

            # If you are using multiple nodes, we assume you are using all the GPUs.
            # On single node, a user can be using only a few GPUs of the node.
            if torch.cuda.is_available() and num_nodes > 1 and world_size % torch.cuda.device_count() != 0:
                raise RuntimeError("The world size should be divisible by the number of GPUs.")
        else:
            world_size = None
            global_rank = 0
            num_nodes = 1

        if world_size is None or world_size == -1:
            world_size = 1

        world_size = int(os.environ.get("WORLD_SIZE", world_size))
        global_rank = int(os.environ.get("GLOBAL_RANK", global_rank))
        num_nodes = int(os.environ.get("NNODES", num_nodes))

        return cls(world_size=world_size, global_rank=global_rank, num_nodes=num_nodes)

    @classmethod
    def _instantiate_in_map_or_optimize(cls) -> "_DistributedEnv":
        global_rank = int(os.getenv("DATA_OPTIMIZER_GLOBAL_RANK", "0"))
        num_workers = int(os.getenv("DATA_OPTIMIZER_NUM_WORKERS", "0"))
        num_nodes = int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 1))
        return cls(world_size=num_workers * num_nodes, global_rank=int(global_rank), num_nodes=num_nodes)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"world_size={self.world_size}, "
            f"global_rank={self.global_rank}, "
            f"num_nodes={self.num_nodes})"
        )

    def __str__(self) -> str:
        return repr(self)


class _WorkerEnv:
    """Contains the environment for the current dataloader within the current training process.

    Args:
        world_size: The number of dataloader workers for the current training process
        rank: The rank of the current worker within the number of workers

    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    @classmethod
    def detect(cls, get_worker_info_fn: Optional[Callable] = None) -> "_WorkerEnv":
        """Automatically detects the number of workers and the current rank.

        .. note::
            This only works reliably within a dataloader worker as otherwise the necessary information won't be present.
            In such a case it will default to 1 worker

        """
        get_worker_info = get_worker_info_fn or torch_get_worker_info
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        current_worker_rank = worker_info.id if worker_info is not None else 0

        return cls(world_size=num_workers, rank=current_worker_rank)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(world_size: {self.world_size}, rank: {self.rank})"

    def __str__(self) -> str:
        return repr(self)


class Environment:
    """Contains the compute environment. If not passed, will try to detect.

    Args:
        dist_env: The distributed environment (distributed worldsize and global rank)
        worker_env: The worker environment (number of workers, worker rank)

    """

    def __init__(self, dist_env: Optional[_DistributedEnv], worker_env: Optional[_WorkerEnv]):
        self.worker_env = worker_env
        self.dist_env = dist_env

    @classmethod
    def from_args(
        cls,
        dist_world_size: int,
        global_rank: int,
        num_workers: int,
        current_worker_rank: int,
    ) -> "Environment":
        """Generates the Environment class by already given arguments instead of detecting them.

        Args:
            dist_world_size: The world-size used for distributed training (=total number of distributed processes)
            global_rank: The distributed global rank of the current process
            num_workers: The number of workers per distributed training process
            current_worker_rank: The rank of the current worker within the number of workers of
                the current training process

        """
        num_nodes = (dist_world_size // torch.cuda.device_count()) if torch.cuda.is_available() else 1
        dist_env = _DistributedEnv(dist_world_size, global_rank, num_nodes)
        worker_env = _WorkerEnv(num_workers, current_worker_rank)
        return cls(dist_env=dist_env, worker_env=worker_env)

    @property
    def num_shards(self) -> int:
        """Returns the total number of shards.

        .. note::
            This may not be accurate in a non-dataloader-worker process like the main training process
            as it doesn't necessarily know about the number of dataloader workers.

        """
        assert self.worker_env is not None
        assert self.dist_env is not None
        return self.worker_env.world_size * self.dist_env.world_size

    @property
    def shard_rank(self) -> int:
        """Returns the rank of the current process wrt. the total number of shards.

        .. note::
            This may not be accurate in a non-dataloader-worker process like the main training process as it
            doesn't necessarily know about the number of dataloader workers.

        """
        assert self.worker_env is not None
        assert self.dist_env is not None
        return self.dist_env.global_rank * self.worker_env.world_size + self.worker_env.rank

    def __repr__(self) -> str:
        dist_env_repr = repr(self.dist_env)
        worker_env_repr = repr(self.worker_env)

        return (
            f"{self.__class__.__name__}(\n\tdist_env: {dist_env_repr},\n\tworker_env: "
            + f"{worker_env_repr}\n\tnum_shards: {self.num_shards},\n\tshard_rank: {self.shard_rank})"
        )

    def __str__(self) -> str:
        return repr(self)


def _is_in_dataloader_worker() -> bool:
    return torch_get_worker_info() is not None


def _is_in_map_or_optimize() -> bool:
    return os.getenv("DATA_OPTIMIZER_GLOBAL_RANK") is not None
