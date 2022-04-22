import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.max_len = max_len

    def __getitem__(self, index):
        tokens = self.tokenizer(
            text=self.texts[index],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "ids": tokens["input_ids"].flatten(),
            "mask": tokens["attention_mask"].flatten(),
            "labels": self.labels[index],
        }

    def __len__(self):
        return len(self.texts)
