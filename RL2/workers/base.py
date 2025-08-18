import math
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import transformers
from RL2.utils.parallelism import prepare_tp_model, prepare_dp_model
from RL2.utils.sequences import (
    pad_tensor_dict_to_multiple_of,
    pack_tensor_dicts_to_minibatches,
    split_minibatches_into_tensor_dicts,
    resume_order_of_tensor_dicts
)
from RL2.utils.comm import (
    split_and_scatter_list,
    boardcast_list,
    gather_and_concat_list
)
from RL2.utils.offloading import load_model_to_device, load_optimizer_to_device

class Worker:

    def __init__(self, config, train: bool):

        self.config = config
        self.train = train

        self.prepare_device_mesh()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % (self.config.ddp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by ddp_size {self.config.ddp_size} * tp_size {self.config.tp_size}."
        self.fsdp_size = world_size // (self.config.ddp_size * self.config.tp_size)
        self.model_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("ddp", "fsdp", "tp"),
            mesh_shape=(self.config.ddp_size, self.fsdp_size, self.config.tp_size)
        )

        assert world_size % (self.config.sp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by sp_size {self.config.sp_size} * tp_size {self.config.tp_size}."
        self.dp_size = world_size // (self.config.sp_size * self.config.tp_size)
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "sp", "tp"),
            mesh_shape=(self.dp_size, self.config.sp_size, self.config.tp_size)
        )

    def prepare_model_optimizer(self):

        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

        self.model = prepare_dp_model(
            self.model, self.model_device_mesh["ddp", "fsdp"]
        )

        if self.train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        load_model_to_device(self, "cpu")

    def scatter_and_pack_tensor_dicts(
        self, tensor_dicts, pack_minibatches=False, pair=False
    ):

        if pack_minibatches:
            # Pack minibatches into multiple batches, where each batch is 
            # used for an update and contains multiple minibatches.
            if dist.get_rank() == 0:
                assert len(tensor_dicts) >= self.config.update_per_rollout, \
                    f"The number of trajectories {len(tensor_dicts)} is less than the number of updates {self.config.update_per_rollout}."
                bsz = math.ceil(
                    len(tensor_dicts) / self.config.update_per_rollout
                )
                return [
                    self.scatter_and_pack_tensor_dicts(
                        tensor_dicts[update * bsz:(update + 1) * bsz]
                    )
                    for update in range(self.config.update_per_rollout)
                ]
            else:
                return [
                    self.scatter_and_pack_tensor_dicts(None)
                    for _ in range(self.config.update_per_rollout)
                ]

        if self.device_mesh["tp"].get_local_rank() == 0:
            if self.device_mesh["sp"].get_local_rank() == 0:
                if self.device_mesh["dp"].get_local_rank() == 0:
                    # We use ZigZag Ring Attention to partition sequences, where 
                    # the length of each sequence needs to be multiple of 2 * 
                    # sp size and each rank sequentially get the head and tail.
                    # See https://zhuanlan.zhihu.com/p/683714620.
                    tensor_dicts = [
                        pad_tensor_dict_to_multiple_of(
                            td, 2 * self.device_mesh["sp"].size()
                        )
                        for td in tensor_dicts
                    ]
                    minibatches = pack_tensor_dicts_to_minibatches(
                        self, tensor_dicts, pair
                    )
                minibatches = split_and_scatter_list(
                    minibatches
                    if self.device_mesh["dp"].get_local_rank() == 0
                    else None,
                    self.device_mesh["dp"]
                )
            minibatches = boardcast_list(
                minibatches
                if self.device_mesh["sp"].get_local_rank() == 0
                else None,
                self.device_mesh["sp"]
            )
        minibatches = boardcast_list(
            minibatches
            if self.device_mesh["tp"].get_local_rank() == 0
            else None,
            self.device_mesh["tp"]
        )
        return [
            {
                k: v.to(torch.cuda.current_device())
                for k, v in minibatch.items()
            }
            for minibatch in minibatches
        ]

    def unpack_and_gather_tensor_dicts(self, minibatches):
        
        tensor_dicts = split_minibatches_into_tensor_dicts(minibatches)
        tensor_dicts = gather_and_concat_list(
            tensor_dicts, self.device_mesh["dp"]
        )
        if dist.get_rank() == 0:
            return resume_order_of_tensor_dicts(tensor_dicts)
            
    def backward(self, loss):
        # https://github.com/ChenmienTan/RL2/issues/11
        (self.dp_size * self.config.sp_size * loss).backward()
    
    def optimizer_step(self):

        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )

        load_optimizer_to_device(
            self, torch.cuda.current_device()
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        load_optimizer_to_device(self, "cpu")

        return grad_norm.item()