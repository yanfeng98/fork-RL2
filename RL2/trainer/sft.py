import hydra
from collections import defaultdict
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import SFTDataset, get_dataloader
from RL2.workers import Actor
from RL2.utils.sequences import count_total
from RL2.utils.functions import aggregate_values
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.checkpointing import load_ckpt, save_ckpt, save_model
from RL2.utils.logging import progress_bar, time_logger, gather_and_log


class SFTTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = Actor(config.actor, True)
        dataset = SFTDataset(
            config.data.path, self.actor.tokenizer, config.data.max_length
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.actor.scheduler = self.prepare_scheduler(self.actor)

    @time_logger("update_actor")
    def update_actor(self, tensor_dicts, step):

        minibatches = self.actor.scatter_and_pack_tensor_dicts(tensor_dicts)
        total_actions = count_total(
            minibatches, "action_mask", self.actor.device_mesh["dp"]
        )
        metrics = defaultdict(list)
        for minibatch in progress_bar(
            minibatches, desc="Update actor"
        ):
            logps = self.actor.forward(minibatch)
            loss = aggregate_values(
                - logps,
                minibatch,
                self.actor.config.agg_mode,
                total_actions,
                self.config.data.batch_size
            )
            self.actor.backward(loss)
            metrics["loss"].append(loss.item())

        grad_norm = self.actor.optimizer_step()
        metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, self.actor.device_mesh["dp"], step)

    def train(self):

        step = load_ckpt(self, (self.actor,))
        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            for tensor_dicts in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1
                self.update_actor(tensor_dicts, step)
                save_ckpt(self, (self.actor,), step)
        save_model(self, self.actor)


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()