from torch import nn
import torch
from GEC_Generator import Generator
from GEC_Discriminator import Discriminator


class GECgan():
    def __init__(self,bert_path, num_class, embedding_size, batch_size, vocab, loss_type, device='cpu', dropout=0.1) -> None:
        self.generator = Generator(bert_path, num_class, embedding_size, batch_size, vocab, loss_type, device='cpu', dropout=0.1)
        self.discriminator = Discriminator(bert_path, num_class, embedding_size, dropout=0.1)
        self.loss_fn = nn.BCELoss()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
