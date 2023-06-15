import json
import torch
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PAD = '<PAD>'
UNK = '<UNK>'

label_dic = {'positive': 1, 'negative': 2, 'neutral': 0}

def read_json(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as f:
      anslist = json.load(f)
    return anslist

def build_dataset(file_path, mode, word2idx, padding_size = 32):
    assert mode in ['train', 'test']
    res = read_json(file_path)
    text, labels = [], []
    for dic in tqdm(res):
        corpus, label = dic['content'], dic['label']
        if mode == 'train':
            label = label_dic[label]
        words = list(jieba.cut(corpus))
        if len(words) < padding_size:
            words += [PAD] * (padding_size - len(words))
        else:
            words = words[: padding_size]
        unk_idx = word2idx[UNK]
        idxs = [word2idx.get(word, unk_idx) for word in words]
        if mode == 'train':
            text.append(idxs)
            labels.append(label)
        else:
            text.append(idxs)
    if mode == 'train':    
        return text, labels
    else:
        return text

class Mydataset(Dataset):
    def __init__(self, file_path, mode, word2idx):
        if mode == 'train':
            self.x, self.y = build_dataset(file_path, mode, word2idx)
        else:
            self.x = build_dataset(file_path, mode, word2idx)
        self.mode = mode
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.len = len(self.x)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.x[index], self.y[index]
        else:
            return self.x[index]
