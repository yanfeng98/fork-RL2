import copy
from RL2.datasets.base import BaseDataset, load_dataset

class RLDataset(BaseDataset):
    
    def __init__(self, config, tokenizer):

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = load_dataset(config.path)

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = ex["prompt"]
        answer = ex["answer"]

        return {
            "prompt": prompt,
            "answer": answer
        }

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]