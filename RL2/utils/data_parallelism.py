import functools
from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import PreTrainedModel

def prepare_dp_model(model: PreTrainedModel, device_mesh: dist.device_mesh.DeviceMesh) -> FSDP:

    def get_module_cls_from_name(name: str) -> type:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__

    transformer_layer_cls: set[type] = {
        get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy: Callable[[torch.nn.Module, bool, int], bool] = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )

    mixed_precision: MixedPrecision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=device_mesh,
        device_id=torch.cuda.current_device()
    )