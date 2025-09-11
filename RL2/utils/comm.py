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

def split_and_scatter_list(lst: list[dict[str, torch.Tensor]], device_mesh: dist.DeviceMesh) -> list[dict[str, torch.Tensor]]:

    if device_mesh.get_local_rank() == 0:
        length_per_dp: int = math.ceil(len(lst) / device_mesh.size())
    lists: list[list[dict[str, torch.Tensor]]] = [
        lst[rank * length_per_dp:(rank + 1) * length_per_dp]
        if device_mesh.get_local_rank() == 0 else None
        for rank in range(device_mesh.size())
    ]
    lst: list[dict[str, torch.Tensor]] = [None]
    dist.scatter_object_list(
        lst,
        lists,
        group=device_mesh.get_group(),
        group_src=0
    )
    return lst[0]

def boardcast_list(lst: list[dict[str, torch.Tensor]], device_mesh: dist.DeviceMesh) -> list[dict[str, torch.Tensor]]:

    kwargs: dict[str, object] = {
        "group": device_mesh.get_group(),
        "group_src": 0
    }
    length: int = torch.LongTensor([
        len(lst)
        if device_mesh.get_local_rank() == 0
        else 0
    ]).to(torch.cuda.current_device())
    dist.broadcast(length, **kwargs)
    if device_mesh.get_local_rank() != 0:
        lst: list[dict[str, torch.Tensor]] = length.item() * [None]
    dist.broadcast_object_list(lst, **kwargs)
    return lst

def gather_and_concat_list(lst, device_mesh):

    lists = (
        device_mesh.size() * [None]
        if device_mesh.get_local_rank() == 0
        else None
    )
    dist.gather_object(
        lst,
        lists,
        group=device_mesh.get_group(),
        group_dst=0
    )
    return sum(lists, []) if device_mesh.get_local_rank() == 0 else None