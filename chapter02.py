# chapter02.py
# 第二章：处理文本数据
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

import tiktoken

class GPTDatasetV2(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        
        full_length = len(token_ids) - 1
        
        if stride >= max_length:
            num_samples = (full_length // max_length) + (1 if full_length % max_length else 0)
        else:
            num_samples = (full_length - max_length) // stride + 1
        
        for i in range(num_samples):
            start_idx = i * stride
            end_idx = start_idx + max_length
            
            if end_idx <= len(token_ids):
                input_chunk = token_ids[start_idx:end_idx]
                target_chunk = token_ids[start_idx + 1:end_idx + 1]
                
                if len(input_chunk) == max_length and len(target_chunk) == max_length:
                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v2(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, 
                         num_workers=0, infinite=False):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV2(txt, tokenizer, max_length, stride)
    
    if infinite:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        from itertools import cycle
        return cycle(dataloader)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        return dataloader
    
    
# chapter02.py
# 第二章和第五章：数据处理、训练工具函数和文本生成
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

# ==================== 数据处理部分 ====================


# ==================== 文本转换工具函数 ====================

def text_to_token_ids(text, tokenizer):
    """将文本转换为token ID张量"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded, dtype=torch.long)