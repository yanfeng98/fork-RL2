import torch
import torch.distributed as dist

def compute_logsumexp(logits: torch.Tensor, device_mesh: dist.device_mesh.DeviceMesh, chunk_size: int = 1024) -> torch.Tensor:

    logsumexps: list[torch.Tensor] = []
    for start in range(0, logits.shape[1], chunk_size):
        # batch_size, chunk
        logsumexp: torch.Tensor = torch.logsumexp(
            logits[:, start:start + chunk_size], -1
        )
        logsumexps.append(logsumexp)
    # batch_size, seq_len
    logsumexp: torch.Tensor = torch.cat(logsumexps, -1)

    logsumexps: list[torch.Tensor] = [
        torch.zeros_like(logsumexp)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        logsumexps,
        logsumexp,
        group=device_mesh.get_group()
    )
    logsumexps[device_mesh.get_local_rank()] = logsumexp # necessary to retain grad
    
    logsumexps: torch.Tensor = torch.cat([
        logsumexp.unsqueeze(-1) for logsumexp in logsumexps
    ], -1) # batch_size, seq_len, tp_size

    return torch.logsumexp(logsumexps, -1)

def gather_action_logits(logits: torch.Tensor, actions: torch.Tensor, device_mesh: dist.device_mesh.DeviceMesh) -> torch.Tensor:

    rank: int = device_mesh.get_local_rank()
    start_idx: int = rank * logits.shape[-1]
    end_idx: int = (rank + 1) * logits.shape[-1]

    # 1, seq_len
    local_mask: torch.Tensor = (actions >= start_idx) & (actions < end_idx)
    # 1, seq_len
    local_actions: torch.Tensor = torch.where(
        local_mask, actions - start_idx, 0
    )

    # 1, seq_len
    action_logits: torch.Tensor = torch.where(
        local_mask,
        torch.gather(
            logits,
            dim=-1,
            index=local_actions.unsqueeze(-1) # 1, seq_len, 1
        ).squeeze(-1),
        0.0
    )

    return differentiable_all_reduce(action_logits, device_mesh)

def differentiable_all_reduce(tensor: torch.Tensor, device_mesh: dist.device_mesh.DeviceMesh) -> torch.Tensor:

    detached_tensor: torch.Tensor = tensor.detach()
    dist.all_reduce(
        detached_tensor,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return tensor + detached_tensor - tensor.detach()

def aggregate_values(
    tensor: torch.Tensor,
    action_mask: torch.Tensor,
    avg_level: str,
    total_actions: int,
    total_sequences: int
) -> torch.Tensor:
    
    if isinstance(tensor, tuple):
        return tuple(
            aggregate_values(
                t,
                action_mask,
                avg_level,
                total_actions,
                total_sequences
            )
            for t in tensor
        )

    if avg_level == "token":
        return tensor.sum() / total_actions
    elif avg_level == "sequence":
        return (
            tensor.sum(-1) / (
                action_mask.sum(-1) + torch.finfo(tensor.dtype).eps
            )
        ).sum() / total_sequences
    else:
        raise NotImplementedError

def compute_entropy(logits: torch.Tensor, logsumexp: torch.Tensor, device_mesh: dist.device_mesh.DeviceMesh):

    probs: torch.Tensor = torch.exp(logits - logsumexp.unsqueeze(-1))
    return logsumexp - differentiable_all_reduce(
        (probs * logits).sum(-1), device_mesh
    )