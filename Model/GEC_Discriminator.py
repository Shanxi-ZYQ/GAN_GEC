import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,bert_path, num_class, embedding_size, dropout=0.1):
        super(Discriminator, self).__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = dropout
        #bert的分类器
        self.fc = nn.Linear(self.embedding_size, 1)

    def forward(self, input_ids,attention_mask,token_type_ids):
        #bert的encoder
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #bert的encoder的输出
        sequence_representation = output.pooler_output
        #dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)
        #bert的分类器
        sequence_representation = self.fc(sequence_representation)
        #softmax
        probs = torch.softmax(sequence_representation, -1)
        return probs
                                        