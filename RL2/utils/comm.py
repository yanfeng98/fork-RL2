import os
import math
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)

def split_and_scatter_list(lst, device_mesh):

    if device_mesh.get_local_rank() == 0:
        data_per_dp = math.ceil(len(lst) / device_mesh.size())
    lists = [
        lst[rank * data_per_dp:(rank + 1) * data_per_dp]
        if device_mesh.get_local_rank() == 0 else None
        for rank in range(device_mesh.size())
    ]
    lst = [None]
    dist.scatter_object_list(
        lst,
        lists,
        group=device_mesh.get_group(),
        group_src=0
    )
    return lst[0]

def gather_and_concat_list(lst, device_mesh):

    lists = [None] * device_mesh.size() if device_mesh.get_local_rank() == 0 else None
    dist.gather_object(
        lst,
        lists,
        group=device_mesh.get_group(),
        group_dst=0
    )

    return sum(lists, []) if device_mesh.get_local_rank() == 0 else None