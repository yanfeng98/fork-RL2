import os
import functools
from typing import Optional, Union, Any

import torch
import torch.distributed as dist

import transformers
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import (
    is_flash_attn_greater_or_equal_2_10
)

from ring_flash_attn.llama3_flash_attn_varlen import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_prepare_cu_seqlens
)
from ring_flash_attn.adapters.hf_adapter import flash_attention_forward

from RL2.workers.base import Worker

DATA_PARAMS: dict[str, Any] = {}

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_k: Optional[torch.Tensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    **kwargs
):
    use_sliding_windows = (
        sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs: dict[str, Any] = (
        {"window_size": (sliding_window, sliding_window)}
        if use_sliding_windows
        else {}
    )

    if is_flash_attn_greater_or_equal_2_10:
        if deterministic is None:
            deterministic = (
                os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
            )

    flash_kwargs["deterministic"] = deterministic
    flash_kwargs["group"] = DATA_PARAMS["group"]

    return llama3_flash_attn_varlen_func(
        query_states.squeeze(0),
        key_states.squeeze(0),
        value_states.squeeze(0),
        cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
        cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
        max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
        max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
        heads_k_stride=1,
        local_k_slice=DATA_PARAMS["local_k_slice"],
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=True,
        **flash_kwargs
    )

transformers.modeling_flash_attention_utils._flash_attention_forward = _flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def sequence_parallelism_manager(func):

    @functools.wraps(func)
    def forward_with_sequence_parallelism(
        worker: Worker, minibatch: dict[str, torch.Tensor], *args, **kwargs
    ):
        # batch_size, seq_len
        shape: torch.Size = minibatch["states"].shape
        # batch_size
        seq_lens: torch.Tensor = minibatch["eos_mask"].argmax(-1) + 1
        # 1, seq_len
        minibatch: dict[str, torch.Tensor] = {
            k: torch.cat([
                seq[:seq_len] for seq, seq_len in zip(v, seq_lens)
            ]).unsqueeze(0)
            for k, v in minibatch.items()
        }

        multiple_of: int = worker.device_mesh["sp"].size() * worker.device_mesh["tp"].size()

        if sum(seq_lens) % multiple_of != 0:
            pad_tokens: int = multiple_of - sum(seq_lens) % multiple_of
            
            seq_lens: torch.Tensor = torch.cat((
                seq_lens,
                torch.LongTensor([pad_tokens]).to(torch.cuda.current_device())
            ))
            minibatch: dict[str, torch.Tensor] = {
                k: torch.cat((
                    v,
                    torch.zeros((1, pad_tokens), dtype=v.dtype, device=v.device)
                ), -1)
                for k, v in minibatch.items()
            }

        # batch_size + 1
        cu_seqlens: torch.Tensor = torch.cumsum(
            torch.cat((
                torch.LongTensor([0]).to(torch.cuda.current_device()),
                seq_lens
            )),
            dim=0,
            dtype=torch.int32
        )
        rank: int = worker.device_mesh["sp"].get_local_rank()
        world_size: int = worker.device_mesh["sp"].size()
        
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens,
            True,
            rank,
            world_size
        )
        
        DATA_PARAMS.update({
            "group": worker.device_mesh["sp"].get_group(),
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "local_k_slice": local_k_slice,
        })
        
        # 1, seq_len
        minibatch: dict[str, torch.Tensor] = {
            k: torch.chunk(v, world_size, dim=-1)[rank]
            for k, v in minibatch.items()
        }

        output: torch.Tensor = func(worker, minibatch, *args, **kwargs)

        def postprocess(output: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:

            if isinstance(output, tuple):
                return tuple(
                    postprocess(tensor)
                    for tensor in output
                )
            
            tensors: list[torch.Tensor] = [
                torch.zeros_like(output)
                for _ in range(world_size)
            ]
            dist.all_gather(
                tensors,
                output,
                group=worker.device_mesh["sp"].get_group()
            )
            tensors[rank] = output # necessary to retain grad
            tensor: torch.Tensor = torch.cat(tensors, -1).squeeze(0)

            output: torch.Tensor = torch.zeros(shape, device=torch.cuda.current_device())
            for row, start_idx, end_idx in zip(
                range(shape[0]), cu_seqlens[:-1], cu_seqlens[1:]
            ):
                output[row, :end_idx - start_idx] = tensor[start_idx:end_idx]

            return output

        return postprocess(output)
    
    return forward_with_sequence_parallelism