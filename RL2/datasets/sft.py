from RL2.datasets import BaseDataset, tokenize_messages

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):

        messages = self.dataset[idx]["messages"]
        tensor_dict = tokenize_messages(self.tokenizer, messages)
        return {k: v[:self.max_length] for k, v in tensor_dict.items()}
    
    def collate_fn(self, batch):
        return list(batch)