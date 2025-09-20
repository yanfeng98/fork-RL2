import os
from omegaconf import DictConfig

import datasets
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader

class BaseDataset(Dataset):
    
    def __init__(self, config: DictConfig, tokenizer: Qwen2Tokenizer):

        self.config: DictConfig = config
        self.dataset: Dataset = (
            load_dataset(config.path) if config.path else
            [{} for _ in range(config.prompts_per_rollout)] # for Gym like environments
        )
        self.tokenizer: Qwen2Tokenizer = tokenizer

    def tokenize_prompt_response(
        self, prompt: str, response: str, rm: bool = False
    ) -> dict[str, torch.LongTensor|torch.Tensor]:
        
        prompt: list[int] = self.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        response: list[int] = self.tokenizer.encode(
            response + self.tokenizer.eos_token,
            add_special_tokens=False
        )
        
        states: list[int] = prompt + response
        actions: list[int] = len(prompt) * [0] + response
        action_mask: list[int] = len(prompt) * [0] + len(response) * [1]
        
        return get_tensor_dict(
            states, actions, action_mask, self.config.max_length, rm
        )

    def tokenize_messages(self, messages: list[dict[str, str]], rm: bool = False) -> dict[str, torch.LongTensor|torch.Tensor]:

        prev_text, states, actions, action_mask = "", [], [], []
        for turn in range(len(messages)):
            is_this_turn_assistant: bool = messages[turn]["role"] == "assistant"
            is_next_turn_assistant: bool = turn + 1 < len(messages) and messages[turn + 1]["role"] == "assistant"

            text: str = self.tokenizer.apply_chat_template(
                messages[:turn + 1],
                add_generation_prompt=is_next_turn_assistant,
                tokenize=False
            )
            assert text[:len(prev_text)] == prev_text
            state: list[int] = self.tokenizer.encode(
                text[len(prev_text):], add_special_tokens=False
            )
            states.extend(state)
            actions.extend(
                state
                if is_this_turn_assistant
                else len(state) * [0]
            )
            action_mask.extend(len(state) * [is_this_turn_assistant])
            prev_text: str = text

        return get_tensor_dict(
            states, actions, action_mask, self.config.max_length, rm
        )

    def __len__(self):
        return len(self.dataset)
    
def load_dataset(data_path: str) -> Dataset:

    if "@" in data_path:
        split, data_path = data_path.split("@")
    else:
        split: str = "train"
    
    ext: str = os.path.splitext(data_path)[-1].strip(".")
    if ext in ["json", "jsonl", "csv", "parquet", "arrow"]:
        if ext == "jsonl":
            ext = "json"
        return datasets.load_dataset(ext, data_files=data_path, split=split)
    else:
        return datasets.load_dataset(data_path, split=split)

def get_tensor_dict(
    states: list[int],
    actions: list[int],
    action_mask: list[bool],
    max_length: int = None,
    rm: bool = False
) -> dict[str, torch.LongTensor|torch.Tensor]:

    if not rm:
        states = states[:-1]
        actions = actions[1:]
        action_mask = action_mask[1:]

    if max_length:
        states = states[:max_length]
        actions = actions[:max_length]
        action_mask = action_mask[:max_length]

    tensor_dict: dict[str, torch.LongTensor|torch.Tensor] = {
        "states": torch.LongTensor(states),
        "eos_mask": torch.LongTensor((len(states) - 1) * [0] + [1]),
        "position_ids": torch.arange(len(states))
    }
    if rm:
        tensor_dict["action_mask"] = torch.LongTensor(
            (len(states) - 1) * [0] + [1]
        )
    else:
        tensor_dict["actions"] = torch.LongTensor(actions)
        tensor_dict["action_mask"] = torch.LongTensor(action_mask)

    return tensor_dict

def pack_tensor_dicts(tensor_dicts: list[dict[str, torch.LongTensor|torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        k: pad_sequence(
            [tensor_dict[k] for tensor_dict in tensor_dicts], True
        )
        for k in tensor_dicts[0].keys()
    }

def get_dataloader(dataset: Dataset, batch_size: int) -> StatefulDataLoader:
    return StatefulDataLoader(
        dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.collate_fn
    )