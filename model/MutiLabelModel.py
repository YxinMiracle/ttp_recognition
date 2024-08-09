from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel


class MutiLabelModel(nn.Module):
    def __init__(self, args, n_classes):
        super(MutiLabelModel, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=self.args.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, x: List[str]):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           truncation=True,
                           max_length=512
                           ).to(self.device)
        x = self.bert(**x)[0]
        cls_embeddings = x[:, 0, :]  # 取每个样本的第一个token（[CLS]）的输出
        output = self.drop(cls_embeddings)
        return self.out(output)
